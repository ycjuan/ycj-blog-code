#include "utils/util.hpp"
#include "worker_cow.hpp"

#include <vector>

__global__ void kn_scatter(T_EMB* d_dst, const EmbElement* d_elements, int embDim, int numElements)
{
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= numElements)
    {
        return;
    }
    d_dst[d_elements[t].rowIdx * embDim + t % embDim] = d_elements[t].val;
}

__global__ void kn_scatter(float* d_dst, const ScalarElement* d_elements, int numElements)
{
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= numElements)
    {
        return;
    }
    d_dst[d_elements[t].rowIdx] = d_elements[t].val;
}

__global__ void kn_setDirty(char* d_dirty, const int* d_rowIdx, int numElements, char val)
{
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= numElements)
    {
        return;
    }
    d_dirty[d_rowIdx[t]] = val;
}

// ---- WorkerCopyOnWrite ----

WorkerCopyOnWrite::WorkerCopyOnWrite(int maxNumDocs, int embDim)
    : Worker(maxNumDocs, embDim)
{
}

void WorkerCopyOnWrite::upsertDocs(const std::vector<long>& v_docId, const std::vector<std::vector<T_EMB>>& v2_embData)
{
    // --- resolve: always allocate new rowIdx (copy-on-write) ---
    CopyOnWriteUpsertData upsertData;
    {
        // --- lock: protect docId<>rowIdx map ---
        std::lock_guard<std::mutex> lock(m_writeMutex);
        upsertData = resolveAndBuildEmbElementsCopyOnWrite(v_docId, v2_embData);
    }

    if (upsertData.v_embElement.empty())
    {
        return;
    }

    const int kBlockSize = 256;
    const int numDocs = (int)v_docId.size();

    // d_element and d_newRowIdx at function scope — outlive all kernel launches
    CudaDeviceArray<EmbElement> d_element(upsertData.v_embElement.size(), "EmbElements");
    CudaDeviceArray<int> d_newRowIdx(numDocs, "newRowIdx");

    // --- H2D: emb data and new rowIdxs ---
    {
        CHECK_CUDA(cudaMemcpyAsync(d_element.data(),
                                   upsertData.v_embElement.data(),
                                   upsertData.v_embElement.size() * sizeof(EmbElement),
                                   cudaMemcpyHostToDevice,
                                   m_writeStream.get()));
        CHECK_CUDA(cudaMemcpyAsync(d_newRowIdx.data(),
                                   upsertData.v_newRowIdx.data(),
                                   numDocs * sizeof(int),
                                   cudaMemcpyHostToDevice,
                                   m_writeStream.get()));
        CHECK_CUDA(cudaStreamSynchronize(m_writeStream.get()));
    }

    // --- scatter emb to new rowIdxs ---
    {
        int scatterGridSize = (upsertData.v_embElement.size() + kBlockSize - 1) / kBlockSize;
        kn_scatter<<<scatterGridSize, kBlockSize, 0, m_writeStream.get()>>>(m_data.data(),
                                                                            d_element.data(),
                                                                            m_embDim,
                                                                            upsertData.v_embElement.size());
        CHECK_CUDA(cudaStreamSynchronize(m_writeStream.get()));
    }

    // --- subclass hook: carry scalars to new rowIdxs before dirty-bit flip ---
    carryScalarsOnUpsert(v_docId, upsertData.v_newRowIdx);

    // --- flip dirty bits: hide old rowIdxs, reveal new rowIdxs ---
    {
        // --- lock: protect dirty bits from concurrent score reads ---
        std::lock_guard<std::mutex> lock(m_readMutex);

        // set dirty=1 on old rowIdxs (hide stale data)
        if (!upsertData.v_oldDirtyRowIdx.empty())
        {
            CudaDeviceArray<int> d_oldDirtyRowIdx(upsertData.v_oldDirtyRowIdx.size(), "oldDirtyRowIdx");
            CHECK_CUDA(cudaMemcpyAsync(d_oldDirtyRowIdx.data(),
                                       upsertData.v_oldDirtyRowIdx.data(),
                                       upsertData.v_oldDirtyRowIdx.size() * sizeof(int),
                                       cudaMemcpyHostToDevice,
                                       m_writeStream.get()));
            int gridSize = (upsertData.v_oldDirtyRowIdx.size() + kBlockSize - 1) / kBlockSize;
            kn_setDirty<<<gridSize, kBlockSize, 0, m_writeStream.get()>>>(m_d_dirty.data(),
                                                                          d_oldDirtyRowIdx.data(),
                                                                          upsertData.v_oldDirtyRowIdx.size(),
                                                                          1);
            CHECK_CUDA(cudaStreamSynchronize(m_writeStream.get()));
        }

        // set dirty=0 on new rowIdxs (make new data visible to scorers)
        {
            int gridSize = (numDocs + kBlockSize - 1) / kBlockSize;
            kn_setDirty<<<gridSize, kBlockSize, 0, m_writeStream.get()>>>(m_d_dirty.data(),
                                                                          d_newRowIdx.data(),
                                                                          numDocs,
                                                                          0);
            CHECK_CUDA(cudaStreamSynchronize(m_writeStream.get()));
        }
    }
}

void WorkerCopyOnWrite::deleteDocs(const std::vector<long>& v_docId)
{
    // --- update maps, collect deleted rowIdxs ---
    std::vector<int> v_deletedRowIdx;
    {
        // --- lock: protect docId<>rowIdx map and scalar map ---
        std::lock_guard<std::mutex> lock(m_writeMutex);
        v_deletedRowIdx = resolveDeletedRowIdxs(v_docId);

        for (long docId : v_docId)
        {
            m_docId2scalar.erase(docId);
        }
    }

    if (v_deletedRowIdx.empty())
    {
        return;
    }

    const int kBlockSize = 256;
    const int gridSize = (v_deletedRowIdx.size() + kBlockSize - 1) / kBlockSize;

    // --- H2D: deleted rowIdxs, set dirty=1 ---
    {
        CudaDeviceArray<int> d_deletedRowIdx(v_deletedRowIdx.size(), "deletedRowIdx");
        CHECK_CUDA(cudaMemcpyAsync(d_deletedRowIdx.data(),
                                   v_deletedRowIdx.data(),
                                   v_deletedRowIdx.size() * sizeof(int),
                                   cudaMemcpyHostToDevice,
                                   m_writeStream.get()));
        {
            // --- lock: protect dirty bits from concurrent score reads ---
            std::lock_guard<std::mutex> lock(m_readMutex);
            kn_setDirty<<<gridSize, kBlockSize, 0, m_writeStream.get()>>>(m_d_dirty.data(),
                                                                          d_deletedRowIdx.data(),
                                                                          v_deletedRowIdx.size(),
                                                                          1);
        }
        CHECK_CUDA(cudaStreamSynchronize(m_writeStream.get()));
    }
}

// ---- WorkerCopyOnWriteEager ----

WorkerCopyOnWriteEager::WorkerCopyOnWriteEager(int maxNumDocs, int embDim)
    : WorkerCopyOnWrite(maxNumDocs, embDim)
{
}

void WorkerCopyOnWriteEager::carryScalarsOnUpsert(const std::vector<long>& v_docId, const std::vector<int>& v_newRowIdx)
{
    std::vector<ScalarElement> v_scalarElement;
    for (int i = 0; i < (int)v_docId.size(); i++)
    {
        auto it = m_docId2scalar.find(v_docId[i]);
        if (it != m_docId2scalar.end())
        {
            v_scalarElement.push_back({ v_newRowIdx[i], it->second });
        }
    }

    if (v_scalarElement.empty())
    {
        return;
    }

    const int kBlockSize = 256;
    CudaDeviceArray<ScalarElement> d_scalarElement(v_scalarElement.size(), "scalarElements");
    CHECK_CUDA(cudaMemcpyAsync(d_scalarElement.data(),
                               v_scalarElement.data(),
                               v_scalarElement.size() * sizeof(ScalarElement),
                               cudaMemcpyHostToDevice,
                               m_writeStream.get()));
    CHECK_CUDA(cudaStreamSynchronize(m_writeStream.get()));
    int gridSize = (v_scalarElement.size() + kBlockSize - 1) / kBlockSize;
    kn_scatter<<<gridSize, kBlockSize, 0, m_writeStream.get()>>>(m_d_scalars.data(),
                                                                 d_scalarElement.data(),
                                                                 v_scalarElement.size());
    CHECK_CUDA(cudaStreamSynchronize(m_writeStream.get()));
}

void WorkerCopyOnWriteEager::updateScalarData(const std::vector<long>& v_docId, const std::vector<float>& v_scalar)
{
    // --- resolve docId -> rowIdx and build scalar elements ---
    std::vector<ScalarElement> v_scalarElement;
    {
        // --- lock: protect docId<>rowIdx map and scalar map ---
        std::lock_guard<std::mutex> lock(m_writeMutex);
        v_scalarElement = resolveScalarElements(v_docId, v_scalar);

        for (int i = 0; i < (int)v_docId.size(); i++)
        {
            m_docId2scalar[v_docId[i]] = v_scalar[i];
        }
    }

    if (v_scalarElement.empty())
    {
        return;
    }

    const int kBlockSize = 256;
    const int numDocs = (int)v_scalarElement.size();

    // --- H2D: scalar data, scatter to GPU immediately ---
    {
        CudaDeviceArray<ScalarElement> d_scalarElement(numDocs, "scalarElements");
        CHECK_CUDA(cudaMemcpyAsync(d_scalarElement.data(),
                                   v_scalarElement.data(),
                                   numDocs * sizeof(ScalarElement),
                                   cudaMemcpyHostToDevice,
                                   m_writeStream.get()));
        CHECK_CUDA(cudaStreamSynchronize(m_writeStream.get()));
        int gridSize = (numDocs + kBlockSize - 1) / kBlockSize;
        kn_scatter<<<gridSize, kBlockSize, 0, m_writeStream.get()>>>(m_d_scalars.data(),
                                                                     d_scalarElement.data(),
                                                                     numDocs);
        CHECK_CUDA(cudaStreamSynchronize(m_writeStream.get()));
    }
}

void WorkerCopyOnWriteEager::score(const std::vector<T_EMB>& v_reqEmb, const std::vector<int>& v_targetRowIdx)
{
    // --- lock: protect dirty bits from concurrent write ops ---
    std::lock_guard<std::mutex> lock(m_readMutex);
    scoreImpl(v_reqEmb, v_targetRowIdx);
}

// ---- WorkerCopyOnWriteLazy ----

WorkerCopyOnWriteLazy::WorkerCopyOnWriteLazy(int maxNumDocs, int embDim)
    : WorkerCopyOnWrite(maxNumDocs, embDim)
{
}

void WorkerCopyOnWriteLazy::updateScalarData(const std::vector<long>& v_docId, const std::vector<float>& v_scalar)
{
    // --- lock: protect scalar map ---
    std::lock_guard<std::mutex> lock(m_writeMutex);

    for (int i = 0; i < (int)v_docId.size(); i++)
    {
        m_docId2scalar[v_docId[i]] = v_scalar[i];
    }
}

void WorkerCopyOnWriteLazy::score(const std::vector<T_EMB>& v_reqEmb, const std::vector<int>& v_targetRowIdx)
{
    // --- lock: protect dirty bits from concurrent write ops ---
    std::lock_guard<std::mutex> lock(m_readMutex);

    // --- sync CPU scalars to GPU for target rows (rowIdx -> docId -> scalar) ---
    {
        const int kBlockSize = 256;

        std::vector<ScalarElement> v_scalarElement;
        v_scalarElement.reserve(v_targetRowIdx.size());
        for (int rowIdx : v_targetRowIdx)
        {
            long docId = m_rowIdx2DocId[rowIdx];
            auto it = m_docId2scalar.find(docId);
            float scalar = (it != m_docId2scalar.end()) ? it->second : 0.0f;
            v_scalarElement.push_back({ rowIdx, scalar });
        }

        CudaDeviceArray<ScalarElement> d_scalarElement(v_scalarElement.size(), "scalarElements");
        CHECK_CUDA(cudaMemcpyAsync(d_scalarElement.data(),
                                   v_scalarElement.data(),
                                   v_scalarElement.size() * sizeof(ScalarElement),
                                   cudaMemcpyHostToDevice,
                                   m_readStream.get()));
        int gridSize = (v_scalarElement.size() + kBlockSize - 1) / kBlockSize;
        kn_scatter<<<gridSize, kBlockSize, 0, m_readStream.get()>>>(m_d_scalars.data(),
                                                                    d_scalarElement.data(),
                                                                    v_scalarElement.size());
        // no sync here — scoreImpl launches on the same stream, ordering is guaranteed
    }

    scoreImpl(v_reqEmb, v_targetRowIdx);
}

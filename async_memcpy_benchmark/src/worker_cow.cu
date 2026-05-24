#include "utils/util.hpp"
#include "worker_cow.hpp"

#include <vector>

static __global__ void kn_setDirty(char* d_dirty, const int* d_rowIdx, int numElements, char val)
{
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= numElements)
    {
        return;
    }
    d_dirty[d_rowIdx[t]] = val;
}

static __global__ void kn_scatter(T_EMB* d_dst, const EmbElement* d_elements, int embDim, int numElements)
{
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= numElements)
    {
        return;
    }
    d_dst[d_elements[t].rowIdx * embDim + t % embDim] = d_elements[t].val;
}

// ---- WorkerCopyOnWrite ----

WorkerCopyOnWrite::WorkerCopyOnWrite(int maxNumDocs, int embDim)
    : Worker(maxNumDocs, embDim)
{
}

CopyOnWriteUpsertData WorkerCopyOnWrite::resolveAndScatterEmb(const std::vector<long>& v_docId,
                                                              const std::vector<std::vector<T_EMB>>& v2_embData,
                                                              CudaDeviceArray<int>& d_newRowIdx)
{
    CopyOnWriteUpsertData upsertData;
    {
        // --- lock: protect docId<>rowIdx map ---
        std::lock_guard<std::mutex> lock(m_writeMutex);
        upsertData = resolveAndBuildEmbElementsCopyOnWrite(v_docId, v2_embData);
    }

    if (upsertData.v_embElement.empty())
    {
        return upsertData;
    }

    const int kBlockSize = 256;
    CudaDeviceArray<EmbElement> d_element(upsertData.v_embElement.size(), "EmbElements");

    // --- H2D: emb data and new rowIdxs ---
    {
        CHECK_CUDA(cudaMemcpyAsync(d_element.data(),
                                   upsertData.v_embElement.data(),
                                   upsertData.v_embElement.size() * sizeof(EmbElement),
                                   cudaMemcpyHostToDevice,
                                   m_writeStream.get()));
        CHECK_CUDA(cudaMemcpyAsync(d_newRowIdx.data(),
                                   upsertData.v_newRowIdx.data(),
                                   upsertData.v_newRowIdx.size() * sizeof(int),
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

    return upsertData;
}

void WorkerCopyOnWrite::commitDirtyBitFlip(const CopyOnWriteUpsertData& upsertData,
                                           const CudaDeviceArray<int>& d_newRowIdx,
                                           int numDocs)
{
    const int kBlockSize = 256;

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
        kn_setDirty<<<gridSize, kBlockSize, 0, m_writeStream.get()>>>(m_d_dirty.data(), d_newRowIdx.data(), numDocs, 0);
        CHECK_CUDA(cudaStreamSynchronize(m_writeStream.get()));
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

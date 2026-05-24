#include "utils/util.hpp"
#include "worker_cow.hpp"

#include <vector>

static __global__ void kn_scatter(float* d_dst, const ScalarElement* d_elements, int numElements)
{
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= numElements)
    {
        return;
    }
    d_dst[d_elements[t].rowIdx] = d_elements[t].val;
}

// ---- WorkerCopyOnWriteEager ----

WorkerCopyOnWriteEager::WorkerCopyOnWriteEager(int maxNumDocs, int embDim)
    : WorkerCopyOnWrite(maxNumDocs, embDim)
{
}

void WorkerCopyOnWriteEager::carryScalarsToNewRowIdx(const std::vector<long>& v_docId,
                                                     const std::vector<int>& v_newRowIdx)
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

void WorkerCopyOnWriteEager::upsertDocs(const std::vector<long>& v_docId,
                                        const std::vector<std::vector<T_EMB>>& v2_embData)
{
    const int numDocs = (int)v_docId.size();
    CudaDeviceArray<int> d_newRowIdx(numDocs, "newRowIdx");

    CopyOnWriteUpsertData upsertData = resolveAndScatterEmb(v_docId, v2_embData, d_newRowIdx);
    if (upsertData.v_embElement.empty())
    {
        return;
    }

    carryScalarsToNewRowIdx(v_docId, upsertData.v_newRowIdx);

    commitDirtyBitFlip(upsertData, d_newRowIdx, numDocs);
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

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

// ---- WorkerCopyOnWriteLazy ----

WorkerCopyOnWriteLazy::WorkerCopyOnWriteLazy(int maxNumDocs, int embDim)
    : WorkerCopyOnWrite(maxNumDocs, embDim)
{
}

void WorkerCopyOnWriteLazy::upsertDocs(const std::vector<long>& v_docId,
                                       const std::vector<std::vector<T_EMB>>& v2_embData)
{
    const int numDocs = (int)v_docId.size();
    CudaDeviceArray<int> d_newRowIdx(numDocs, "newRowIdx");

    CopyOnWriteUpsertData upsertData = resolveAndScatterEmb(v_docId, v2_embData, d_newRowIdx);
    if (upsertData.v_embElement.empty())
    {
        return;
    }

    commitDirtyBitFlip(upsertData, d_newRowIdx, numDocs);
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

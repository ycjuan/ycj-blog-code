#include "utils/util.hpp"
#include "worker_cow.hpp"

#include <vector>

// Scatters scalar values to non-contiguous (row, scalarIdx) locations.
// Each thread writes one float; independent of embedding scatter.
static __global__ void kn_scatter(float* d_dst, const ScalarElement* d_elements, int numScalars, int numElements)
{
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= numElements)
    {
        return;
    }
    d_dst[d_elements[t].rowIdx * numScalars + d_elements[t].scalarIdx] = d_elements[t].val;
}

// ---- WorkerCopyOnWriteLazy ----

WorkerCopyOnWriteLazy::WorkerCopyOnWriteLazy(int maxNumDocs, int embDim, int numScalars)
    : WorkerCopyOnWrite(maxNumDocs, embDim, numScalars)
{
}

void WorkerCopyOnWriteLazy::upsertDocs(const std::vector<long>&               v_docId,
                                       const std::vector<std::vector<T_EMB>>& v2_embData)
{
    const int            numDocs = (int)v_docId.size();
    CudaDeviceArray<int> d_newRowIdx(numDocs, "newRowIdx");

    CopyOnWriteUpsertData upsertData;
    {
        // Hold m_writeMutex across resolveAndScatterEmb to serialize m_writeStream access
        // with concurrent updateScalarData / deleteDocs.
        std::lock_guard<std::mutex> writeLock(m_writeMutex);

        // No scalar handling here — scalars are synced to GPU lazily in score().
        upsertData = resolveAndScatterEmb(v_docId, v2_embData, d_newRowIdx);
        if (upsertData.v_embElement.empty())
        {
            return;
        }
    }

    commitDirtyBitFlip(upsertData, d_newRowIdx, numDocs);
}

void WorkerCopyOnWriteLazy::updateScalarData(const std::vector<long>&               v_docId,
                                             const std::vector<std::vector<float>>& v2_scalar)
{
    // --- lock: protect scalar map ---
    std::lock_guard<std::mutex> lock(m_writeMutex);

    for (int i = 0; i < (int)v_docId.size(); i++)
    {
        m_docId2scalar[v_docId[i]] = v2_scalar[i];
    }
}

void WorkerCopyOnWriteLazy::score(const std::vector<T_EMB>& v_reqEmb,
                                  const std::vector<int>&   v_targetRowIdx,
                                  int                       targetScalarIdx)
{
    // Snapshot scalars from the CPU maps under m_writeMutex before taking m_readMutex.
    // Both m_rowIdx2DocId and m_docId2scalar are written under m_writeMutex; reading
    // them without that lock would be a data race. Lock ordering: m_writeMutex first.
    std::vector<ScalarElement> v_scalarElement;
    {
        std::lock_guard<std::mutex> writeLock(m_writeMutex);
        v_scalarElement.reserve(v_targetRowIdx.size() * m_numScalars);
        for (int rowIdx : v_targetRowIdx)
        {
            long docId = m_rowIdx2DocId[rowIdx];
            auto it    = m_docId2scalar.find(docId);
            for (int j = 0; j < m_numScalars; j++)
            {
                float val = (it != m_docId2scalar.end() && j < (int)it->second.size()) ? it->second[j] : 0.0f;
                v_scalarElement.push_back({ rowIdx, j, val });
            }
        }
    }

    // H2D scalar sync + score kernel must be atomic with respect to dirty-bit flips.
    {
        const int kBlockSize = 256;

        std::lock_guard<std::mutex>    readLock(m_readMutex);
        CudaDeviceArray<ScalarElement> d_scalarElement(v_scalarElement.size(), "scalarElements");
        CHECK_CUDA(cudaMemcpyAsync(d_scalarElement.data(),
                                   v_scalarElement.data(),
                                   v_scalarElement.size() * sizeof(ScalarElement),
                                   cudaMemcpyHostToDevice,
                                   m_readStream.get()));
        int gridSize = (v_scalarElement.size() + kBlockSize - 1) / kBlockSize;
        kn_scatter<<<gridSize, kBlockSize, 0, m_readStream.get()>>>(m_d_scalars.data(),
                                                                    d_scalarElement.data(),
                                                                    m_numScalars,
                                                                    v_scalarElement.size());
        CHECK_CUDA(cudaStreamSynchronize(m_readStream.get()));
        scoreImpl(v_reqEmb, v_targetRowIdx, targetScalarIdx);
    }
}

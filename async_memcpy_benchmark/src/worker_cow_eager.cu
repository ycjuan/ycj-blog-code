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

// ---- WorkerCopyOnWriteEager ----

WorkerCopyOnWriteEager::WorkerCopyOnWriteEager(int maxNumDocs, int embDim, int numScalars)
    : WorkerCopyOnWrite(maxNumDocs, embDim, numScalars)
{
}

// Copies existing CPU scalars for each doc to its new rowIdx on the GPU.
// Must be called after resolveAndScatterEmb (so v_newRowIdx is final) and
// before commitDirtyBitFlip (so scalars are visible when the row is revealed).
void WorkerCopyOnWriteEager::carryScalarsToNewRowIdx(const std::vector<long>& v_docId,
                                                     const std::vector<int>&  v_newRowIdx)
{
    // Build ScalarElements targeting the new rowIdxs from the CPU scalar mirror.
    // Docs with no scalar history are skipped — their GPU slot stays zero-initialized.
    std::vector<ScalarElement> v_scalarElement;
    for (int i = 0; i < (int)v_docId.size(); i++)
    {
        auto it = m_docId2scalar.find(v_docId[i]);
        if (it == m_docId2scalar.end())
        {
            continue;
        }
        for (int j = 0; j < (int)it->second.size(); j++)
        {
            v_scalarElement.push_back({ v_newRowIdx[i], j, it->second[j] });
        }
    }

    if (v_scalarElement.empty())
    {
        return;
    }

    const int                      kBlockSize = 256;
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
                                                                 m_numScalars,
                                                                 v_scalarElement.size());
    CHECK_CUDA(cudaStreamSynchronize(m_writeStream.get()));
}

void WorkerCopyOnWriteEager::upsertDocs(const std::vector<long>&               v_docId,
                                        const std::vector<std::vector<T_EMB>>& v2_embData)
{
    const int            numDocs = (int)v_docId.size();
    CudaDeviceArray<int> d_newRowIdx(numDocs, "newRowIdx");

    CopyOnWriteUpsertData upsertData;
    {
        // Hold m_writeMutex across Phase 1 and 2 to serialize m_writeStream and m_docId2scalar
        // access with concurrent updateScalarData / deleteDocs.
        std::lock_guard<std::mutex> writeLock(m_writeMutex);

        // Phase 1: allocate new rowIdxs, write emb data to GPU.
        upsertData = resolveAndScatterEmb(v_docId, v2_embData, d_newRowIdx);
        if (upsertData.v_embElement.empty())
        {
            return;
        }

        // Phase 2: copy existing scalars to the new rowIdxs so they are current when revealed.
        // New rows are still DIRTY, so concurrent scorers skip them.
        carryScalarsToNewRowIdx(v_docId, upsertData.v_newRowIdx);
    }

    // Phase 3: atomically hide old rows and reveal new rows.
    commitDirtyBitFlip(upsertData, d_newRowIdx, numDocs);
}

void WorkerCopyOnWriteEager::updateScalarData(const std::vector<long>&               v_docId,
                                              const std::vector<std::vector<float>>& v2_scalar)
{
    // Hold m_writeMutex for the entire operation to serialize m_writeStream access
    // with concurrent upsertDocs / deleteDocs.
    std::lock_guard<std::mutex> writeLock(m_writeMutex);

    std::vector<ScalarElement> v_scalarElement = resolveScalarElements(v_docId, v2_scalar);

    // Update CPU mirror so carryScalarsToNewRowIdx sees the latest values on the next upsert.
    for (int i = 0; i < (int)v_docId.size(); i++)
    {
        m_docId2scalar[v_docId[i]] = v2_scalar[i];
    }

    if (v_scalarElement.empty())
    {
        return;
    }

    const int kBlockSize  = 256;
    const int numElements = (int)v_scalarElement.size();

    // --- H2D: scalar data, scatter to GPU immediately ---
    {
        CudaDeviceArray<ScalarElement> d_scalarElement(numElements, "scalarElements");
        CHECK_CUDA(cudaMemcpyAsync(d_scalarElement.data(),
                                   v_scalarElement.data(),
                                   numElements * sizeof(ScalarElement),
                                   cudaMemcpyHostToDevice,
                                   m_writeStream.get()));
        CHECK_CUDA(cudaStreamSynchronize(m_writeStream.get()));
        int gridSize = (numElements + kBlockSize - 1) / kBlockSize;
        kn_scatter<<<gridSize, kBlockSize, 0, m_writeStream.get()>>>(m_d_scalars.data(),
                                                                     d_scalarElement.data(),
                                                                     m_numScalars,
                                                                     numElements);
        CHECK_CUDA(cudaStreamSynchronize(m_writeStream.get()));
    }
}

void WorkerCopyOnWriteEager::score(const std::vector<T_EMB>& v_reqEmb,
                                   const std::vector<int>&   v_targetRowIdx,
                                   int                       targetScalarIdx)
{
    // --- lock: protect dirty bits from concurrent write ops ---
    std::lock_guard<std::mutex> lock(m_readMutex);
    scoreImpl(v_reqEmb, v_targetRowIdx, targetScalarIdx);
}

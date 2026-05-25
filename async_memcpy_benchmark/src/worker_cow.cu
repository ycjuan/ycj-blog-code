#include "utils/util.hpp"
#include "worker_cow.hpp"

#include <vector>

// Sets d_dirty[d_rowIdx[t]] = val for each thread t. Used to atomically hide or
// reveal a batch of rows during the dirty-bit flip.
static __global__ void kn_setDirty(char* d_dirty, const int* d_rowIdx, int numElements, char val)
{
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= numElements)
    {
        return;
    }
    d_dirty[d_rowIdx[t]] = val;
}

// Scatters embedding values to non-contiguous rows. Each thread writes one
// T_EMB value; t % embDim gives the column within the row.
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

WorkerCopyOnWrite::WorkerCopyOnWrite(int maxNumDocs, int embDim, int numScalars)
    : Worker(maxNumDocs, embDim, numScalars)
{
}

CopyOnWriteUpsertData WorkerCopyOnWrite::resolveAndScatterEmb(const std::vector<long>&               v_docId,
                                                              const std::vector<std::vector<T_EMB>>& v2_embData,
                                                              CudaDeviceArray<int>&                  d_newRowIdx)
{
    CopyOnWriteUpsertData upsertData;
    {
        // --- lock: protect docId<>rowIdx map ---
        std::lock_guard<std::mutex> lock(m_writeMutex);

        upsertData.v_embElement.reserve(v_docId.size() * m_embDim);
        upsertData.v_newRowIdx.reserve(v_docId.size());

        for (int i = 0; i < (int)v_docId.size(); i++)
        {
            // Always allocate a fresh row — the new emb is written there while the old
            // row stays visible to concurrent scorers until the dirty-bit flip commits.
            int newRowIdx;
            if (!m_emptyRowIdxSet.empty())
            {
                newRowIdx = *m_emptyRowIdxSet.begin();
                m_emptyRowIdxSet.erase(m_emptyRowIdxSet.begin());
            }
            else
            {
                newRowIdx = m_headRowIdx++;
            }

            auto it = m_docId2rowIdx.find(v_docId[i]);
            if (it != m_docId2rowIdx.end())
            {
                // Existing doc: remap to new row and return old row to free list.
                // The old rowIdx is collected so commitDirtyBitFlip can mark it dirty.
                upsertData.v_oldDirtyRowIdx.push_back(it->second);
                m_emptyRowIdxSet.insert(it->second);
                it->second = newRowIdx;
            }
            else
            {
                m_docId2rowIdx[v_docId[i]] = newRowIdx;
            }
            m_rowIdx2DocId[newRowIdx] = v_docId[i];

            upsertData.v_newRowIdx.push_back(newRowIdx);
            for (int j = 0; j < m_embDim; j++)
            {
                upsertData.v_embElement.push_back({ newRowIdx, v2_embData[i][j] });
            }
        }
    }

    if (upsertData.v_embElement.empty())
    {
        return upsertData;
    }

    const int                   kBlockSize = 256;
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
                                           const CudaDeviceArray<int>&  d_newRowIdx,
                                           int                          numDocs)
{
    const int kBlockSize = 256;

    // --- lock: protect dirty bits from concurrent score reads ---
    std::lock_guard<std::mutex> lock(m_readMutex);

    // Hide old rows first so scorers never see them after the new rows are revealed.
    // Only present for re-upserted (existing) docs; new docs have no old row to hide.
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

    // Reveal new rows; emb (and scalars for Eager) are fully written before this point.
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
    const int gridSize   = (v_deletedRowIdx.size() + kBlockSize - 1) / kBlockSize;

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

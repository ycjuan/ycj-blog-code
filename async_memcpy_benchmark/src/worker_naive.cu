#include "utils/util.hpp"
#include "worker_naive.hpp"

#include <vector>

struct CopyElement
{
    int rowIdx;
    T_EMB val;
};

__global__ void scatterKernel(T_EMB* dst, const CopyElement* elements, int embDim, int numElements)
{
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= numElements)
    {
        return;
    }
    dst[elements[t].rowIdx * embDim + t % embDim] = elements[t].val;
}

WorkerNaive::WorkerNaive(int maxNumDocs, int embDim)
    : Worker(maxNumDocs, embDim)
{
}

void WorkerNaive::updateScalarData(const std::vector<long>& docIds, const std::vector<float>& scalars)
{
    std::lock_guard<std::mutex> lock(m_mutex);
    for (int i = 0; i < (int)docIds.size(); i++)
    {
        auto it = m_docId2rowIdx.find(docIds[i]);
        if (it == m_docId2rowIdx.end())
        {
            continue;
        }
        int rowIdx = it->second;
        CHECK_CUDA(cudaMemcpy(m_d_scalars.data() + rowIdx, &scalars[i], sizeof(float), cudaMemcpyHostToDevice));
    }
}

void WorkerNaive::upsertDoc(const std::vector<long>& v_docIds, const std::vector<std::vector<T_EMB>>& embData2D)
{
    std::lock_guard<std::mutex> lock(m_mutex);
    std::vector<CopyElement> v_elements;
    v_elements.reserve(v_docIds.size() * m_embDim);

    for (int i = 0; i < (int)v_docIds.size(); i++)
    {
        auto it = m_docId2rowIdx.find(v_docIds[i]);
        int rowIdx;
        if (it == m_docId2rowIdx.end())
        {
            if (!m_emptyRowIdxSet.empty())
            {
                rowIdx = *m_emptyRowIdxSet.begin();
                m_emptyRowIdxSet.erase(m_emptyRowIdxSet.begin());
            }
            else
            {
                rowIdx = m_headRowIdx++;
            }
            m_docId2rowIdx[v_docIds[i]] = rowIdx;
            m_rowIdx2DocId[rowIdx] = v_docIds[i];
        }
        else
        {
            rowIdx = it->second;
        }
        for (int j = 0; j < m_embDim; j++)
        {
            v_elements.push_back({ rowIdx, embData2D[i][j] });
        }
    }

    if (v_elements.empty())
    {
        return;
    }

    CudaDeviceArray<CopyElement> d_elements(v_elements.size(), "CopyElements");
    CHECK_CUDA(cudaMemcpyAsync(d_elements.data(),
                               v_elements.data(),
                               v_elements.size() * sizeof(CopyElement),
                               cudaMemcpyHostToDevice,
                               m_writeStream.get()));

    const int kBlockSize = 256;
    const int gridSize = (v_elements.size() + kBlockSize - 1) / kBlockSize;
    scatterKernel<<<gridSize, kBlockSize, 0, m_writeStream.get()>>>(m_data.data(),
                                                                    d_elements.data(),
                                                                    m_embDim,
                                                                    v_elements.size());
    CHECK_CUDA(cudaStreamSynchronize(m_writeStream.get()));
}

void WorkerNaive::deleteDoc(long docId)
{
    std::lock_guard<std::mutex> lock(m_mutex);
    auto it = m_docId2rowIdx.find(docId);
    if (it == m_docId2rowIdx.end())
    {
        return;
    }
    int rowIdx = it->second;
    m_docId2rowIdx.erase(it);
    m_rowIdx2DocId[rowIdx] = -1;
    m_emptyRowIdxSet.insert(rowIdx);
    char dirty = 1;
    CHECK_CUDA(cudaMemcpy(m_d_dirty.data() + rowIdx, &dirty, sizeof(char), cudaMemcpyHostToDevice));
}

void WorkerNaive::score(const std::vector<T_EMB>& reqEmb, const std::vector<int>& targetRowIdxVec)
{
    std::lock_guard<std::mutex> lock(m_mutex);
    scoreImpl(reqEmb, targetRowIdxVec);
}

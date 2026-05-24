#include "utils/util.hpp"
#include "worker_naive.hpp"

#include <vector>

struct CopyElement
{
    int rowIdx;
    T_EMB val;
};

struct ScalarElement
{
    int rowIdx;
    float val;
};

__global__ void scatterScalarKernel(float* dst, const ScalarElement* elements, int numElements)
{
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= numElements)
    {
        return;
    }
    dst[elements[t].rowIdx] = elements[t].val;
}

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

void WorkerNaive::updateScalarData(const std::vector<long>& v_docId, const std::vector<float>& v_scalar)
{
    std::lock_guard<std::mutex> lock(m_mutex);
    std::vector<ScalarElement> v_scalarElement;
    v_scalarElement.reserve(v_docId.size());

    for (int i = 0; i < (int)v_docId.size(); i++)
    {
        auto it = m_docId2rowIdx.find(v_docId[i]);
        if (it == m_docId2rowIdx.end())
        {
            continue;
        }
        v_scalarElement.push_back({ it->second, v_scalar[i] });
    }

    if (v_scalarElement.empty())
    {
        return;
    }

    CudaDeviceArray<ScalarElement> d_scalarElement(v_scalarElement.size(), "scalarElements");
    CHECK_CUDA(cudaMemcpyAsync(d_scalarElement.data(),
                               v_scalarElement.data(),
                               v_scalarElement.size() * sizeof(ScalarElement),
                               cudaMemcpyHostToDevice,
                               m_writeStream.get()));

    const int kBlockSize = 256;
    const int gridSize = (v_scalarElement.size() + kBlockSize - 1) / kBlockSize;
    scatterScalarKernel<<<gridSize, kBlockSize, 0, m_writeStream.get()>>>(m_d_scalars.data(),
                                                                          d_scalarElement.data(),
                                                                          v_scalarElement.size());
    CHECK_CUDA(cudaStreamSynchronize(m_writeStream.get()));
}

void WorkerNaive::upsertDocs(const std::vector<long>& v_docId, const std::vector<std::vector<T_EMB>>& v2_embData)
{
    std::lock_guard<std::mutex> lock(m_mutex);
    std::vector<CopyElement> v_element;
    v_element.reserve(v_docId.size() * m_embDim);

    for (int i = 0; i < (int)v_docId.size(); i++)
    {
        auto it = m_docId2rowIdx.find(v_docId[i]);
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
            m_docId2rowIdx[v_docId[i]] = rowIdx;
            m_rowIdx2DocId[rowIdx] = v_docId[i];
        }
        else
        {
            rowIdx = it->second;
        }
        for (int j = 0; j < m_embDim; j++)
        {
            v_element.push_back({ rowIdx, v2_embData[i][j] });
        }
    }

    if (v_element.empty())
    {
        return;
    }

    CudaDeviceArray<CopyElement> d_elements(v_element.size(), "CopyElements");
    CHECK_CUDA(cudaMemcpyAsync(d_elements.data(),
                               v_element.data(),
                               v_element.size() * sizeof(CopyElement),
                               cudaMemcpyHostToDevice,
                               m_writeStream.get()));

    const int kBlockSize = 256;
    const int gridSize = (v_element.size() + kBlockSize - 1) / kBlockSize;
    scatterKernel<<<gridSize, kBlockSize, 0, m_writeStream.get()>>>(m_data.data(),
                                                                    d_elements.data(),
                                                                    m_embDim,
                                                                    v_element.size());
    CHECK_CUDA(cudaStreamSynchronize(m_writeStream.get()));
}

void WorkerNaive::deleteDocs(const std::vector<long>& v_docId)
{
    std::lock_guard<std::mutex> lock(m_mutex);
    for (long docId : v_docId)
    {
        auto it = m_docId2rowIdx.find(docId);
        if (it == m_docId2rowIdx.end())
        {
            continue;
        }
        int rowIdx = it->second;
        m_docId2rowIdx.erase(it);
        m_rowIdx2DocId[rowIdx] = -1;
        m_emptyRowIdxSet.insert(rowIdx);
        char dirty = 1;
        CHECK_CUDA(cudaMemcpy(m_d_dirty.data() + rowIdx, &dirty, sizeof(char), cudaMemcpyHostToDevice));
    }
}

void WorkerNaive::score(const std::vector<T_EMB>& v_reqEmb, const std::vector<int>& v_targetRowIdx)
{
    std::lock_guard<std::mutex> lock(m_mutex);
    scoreImpl(v_reqEmb, v_targetRowIdx);
}

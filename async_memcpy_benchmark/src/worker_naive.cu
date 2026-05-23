#include "worker_naive.hpp"
#include "utils/util.hpp"

#include <vector>

struct CopyElement
{
    int docIdx;
    T_EMB val;
};

__global__ void scatterKernel(T_EMB* dst, const CopyElement* elements, int embDim, int numElements)
{
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= numElements)
    {
        return;
    }
    dst[elements[t].docIdx * embDim + t % embDim] = elements[t].val;
}

WorkerNaive::WorkerNaive(int maxNumDocs, int embDim)
    : Worker(maxNumDocs, embDim)
{
}

void WorkerNaive::updateScalarData(const std::vector<long>& jobIds, const std::vector<float>& scalars)
{
    for (int i = 0; i < (int)jobIds.size(); i++)
    {
        auto it = m_docId2Idx.find(jobIds[i]);
        if (it == m_docId2Idx.end())
        {
            continue;
        }
        int docIdx = it->second;
        CHECK_CUDA(cudaMemcpy(m_d_scalars.data() + docIdx, &scalars[i], sizeof(float), cudaMemcpyHostToDevice));
    }
}

void WorkerNaive::updateEmbData(const std::vector<long>& v_jobIds, const std::vector<std::vector<T_EMB>>& embData2D)
{
    std::vector<CopyElement> v_elements;
    v_elements.reserve(v_jobIds.size() * m_embDim);

    for (int i = 0; i < (int)v_jobIds.size(); i++)
    {
        auto it = m_docId2Idx.find(v_jobIds[i]);
        if (it == m_docId2Idx.end())
        {
            continue;
        }
        int docIdx = it->second;
        for (int j = 0; j < m_embDim; j++)
        {
            v_elements.push_back({ docIdx, embData2D[i][j] });
        }
    }

    if (v_elements.empty())
    {
        return;
    }

    CudaDeviceArray<CopyElement> d_elements(v_elements.size(), "CopyElements");
    CHECK_CUDA(cudaMemcpyAsync(d_elements.data(), v_elements.data(), v_elements.size() * sizeof(CopyElement), cudaMemcpyHostToDevice, m_stream.get()));

    const int kBlockSize = 256;
    const int gridSize = (v_elements.size() + kBlockSize - 1) / kBlockSize;
    scatterKernel<<<gridSize, kBlockSize, 0, m_stream.get()>>>(m_data.data(), d_elements.data(), m_embDim, v_elements.size());
    CHECK_CUDA(cudaStreamSynchronize(m_stream.get()));
}


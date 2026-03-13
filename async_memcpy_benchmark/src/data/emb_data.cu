#include "data/emb_data.hpp"
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

EmbData::EmbData(int maxNumDocs, int embDim)
    : m_maxNumDocs(maxNumDocs)
    , m_embDim(embDim)
    , m_data(maxNumDocs * embDim, "EmbData")
{
}

const T_EMB* EmbData::data() const { return m_data.data(); }

void EmbData::update(const std::vector<long>& jobIds, const std::vector<std::vector<T_EMB>>& embData2D)
{
    std::vector<CopyElement> hostElements;
    hostElements.reserve(jobIds.size() * m_embDim);

    for (int i = 0; i < (int)jobIds.size(); i++)
    {
        auto it = m_docId2Idx.find(jobIds[i]);
        if (it == m_docId2Idx.end())
        {
            continue;
        }
        int docIdx = it->second;
        for (int j = 0; j < m_embDim; j++)
        {
            hostElements.push_back({ docIdx, embData2D[i][j] });
        }
    }

    if (hostElements.empty())
    {
        return;
    }

    CudaDeviceArray<CopyElement> d_elements(hostElements.size(), "CopyElements");
    CHECK_CUDA(cudaMemcpy(d_elements.data(), hostElements.data(), hostElements.size() * sizeof(CopyElement), cudaMemcpyHostToDevice));

    const int blockSize = 256;
    const int gridSize = (hostElements.size() + blockSize - 1) / blockSize;
    scatterKernel<<<gridSize, blockSize>>>(m_data.data(), d_elements.data(), m_embDim, hostElements.size());
    CHECK_CUDA(cudaGetLastError());
}

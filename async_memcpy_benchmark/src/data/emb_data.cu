#include "data/emb_data.hpp"
#include "utils/util.hpp"

#include <algorithm>
#include <cublas_v2.h>
#include <numeric>
#include <thrust/device_ptr.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>
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
    , m_idxToDocId(maxNumDocs, -1)
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

std::vector<std::vector<long>> EmbData::score(const std::vector<std::vector<T_EMB>>& reqEmb, int k) const
{
    const int numReqs = reqEmb.size();
    const int numDocs = m_docId2Idx.size();

    // Flatten reqEmb to host array and copy to device
    std::vector<T_EMB> hostReqEmb(numReqs * m_embDim);
    for (int i = 0; i < numReqs; i++)
    {
        std::copy(reqEmb[i].begin(), reqEmb[i].end(), hostReqEmb.begin() + i * m_embDim);
    }
    CudaDeviceArray<T_EMB> d_reqEmb(numReqs * m_embDim, "reqEmb");
    CHECK_CUDA(cudaMemcpy(d_reqEmb.data(), hostReqEmb.data(), numReqs * m_embDim * sizeof(T_EMB), cudaMemcpyHostToDevice));

    // GEMM: scores[numReqs x numDocs] = reqEmb[numReqs x embDim] * docEmb[numDocs x embDim]^T
    CudaDeviceArray<float> d_scores(numReqs * numDocs, "scores");
    cublasHandle_t handle;
    cublasCreate(&handle);

    const float alpha = 1.0f;
    const float beta = 0.0f;
    // cuBLAS is column-major. To compute C(M x N) = A(M x K) * B(N x K)^T in row-major:
    // call with (CUBLAS_OP_T, CUBLAS_OP_N, N, M, K, B, K, A, K, C, N)
    cublasGemmEx(
        handle,
        CUBLAS_OP_T, CUBLAS_OP_N,
        numDocs, numReqs, m_embDim,
        &alpha,
        m_data.data(), CUDA_R_16BF, m_embDim,
        d_reqEmb.data(), CUDA_R_16BF, m_embDim,
        &beta,
        d_scores.data(), CUDA_R_32F, numDocs,
        CUBLAS_COMPUTE_32F,
        CUBLAS_GEMM_DEFAULT);

    cublasDestroy(handle);

    // Allocate d_indices[numReqs x numDocs] and initialize each row as [0..numDocs-1]
    CudaDeviceArray<int> d_indices(numReqs * numDocs, "indices");
    for (int i = 0; i < numReqs; i++)
    {
        thrust::sequence(
            thrust::device,
            thrust::device_ptr<int>(d_indices.data() + i * numDocs),
            thrust::device_ptr<int>(d_indices.data() + (i + 1) * numDocs));
    }

    // Sort each row by score descending, carrying indices along
    for (int i = 0; i < numReqs; i++)
    {
        thrust::sort_by_key(
            thrust::device,
            thrust::device_ptr<float>(d_scores.data() + i * numDocs),
            thrust::device_ptr<float>(d_scores.data() + (i + 1) * numDocs),
            thrust::device_ptr<int>(d_indices.data() + i * numDocs),
            thrust::greater<float>());
    }

    // Copy top-k indices per request back to host
    const int topK = std::min(k, numDocs);
    std::vector<int> hostTopIndices(numReqs * topK);
    for (int i = 0; i < numReqs; i++)
    {
        CHECK_CUDA(cudaMemcpy(
            hostTopIndices.data() + i * topK,
            d_indices.data() + i * numDocs,
            topK * sizeof(int),
            cudaMemcpyDeviceToHost));
    }

    // Map indices to docIds
    std::vector<std::vector<long>> results(numReqs);
    for (int i = 0; i < numReqs; i++)
    {
        results[i].resize(topK);
        for (int j = 0; j < topK; j++)
        {
            results[i][j] = m_idxToDocId[hostTopIndices[i * topK + j]];
        }
    }

    return results;
}

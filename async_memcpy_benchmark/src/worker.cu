#include "worker.hpp"
#include "utils/util.hpp"

#include <algorithm>
#include <cublas_v2.h>
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

Worker::Worker(int maxNumDocs, int embDim)
    : m_maxNumDocs(maxNumDocs)
    , m_embDim(embDim)
    , m_data(maxNumDocs * embDim, "Worker")
    , m_idxToDocId(maxNumDocs, -1)
{
}

const T_EMB* Worker::data() const { return m_data.data(); }


std::tuple<int, int, std::vector<int>> Worker::scoreCore(const std::vector<std::vector<T_EMB>>& reqEmb, int k) const
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

    return { numReqs, topK, hostTopIndices };
}

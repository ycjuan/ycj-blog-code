#include "worker.hpp"
#include "utils/util.hpp"

#include <cublas_v2.h>
#include <vector>


Worker::Worker(int maxNumDocs, int embDim)
    : m_maxNumDocs(maxNumDocs)
    , m_embDim(embDim)
    , m_data(maxNumDocs * embDim, "Worker")
    , m_d_scalars(maxNumDocs, "scalars")
    , m_d_scores(maxNumDocs, "scores")
    , m_idxToDocId(maxNumDocs, -1)
{
    CHECK_CUDA(cudaMemset(m_d_scalars.data(), 0, maxNumDocs * sizeof(float)));
}

const T_EMB* Worker::data() const { return m_data.data(); }

void Worker::setScalars(const std::vector<long>& jobIds, const std::vector<float>& scalars)
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

__global__ void applyScalarsKernel(float* scores, const float* scalars, int numReqs, int numDocs)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numReqs * numDocs)
    {
        return;
    }
    int docIdx = idx % numDocs;
    scores[idx] *= scalars[docIdx];
}

void Worker::scoreCore(const std::vector<std::vector<T_EMB>>& reqEmb) const
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
    if (m_d_scores.getArraySize() < (uint64_t)(numReqs * numDocs))
    {
        m_d_scores = CudaDeviceArray<float>(numReqs * numDocs, "scores");
    }
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
        m_d_scores.data(), CUDA_R_32F, numDocs,
        CUBLAS_COMPUTE_32F,
        CUBLAS_GEMM_DEFAULT);

    cublasDestroy(handle);

    const int kBlockSize = 256;
    const int gridSize = (numReqs * numDocs + kBlockSize - 1) / kBlockSize;
    applyScalarsKernel<<<gridSize, kBlockSize>>>(m_d_scores.data(), m_d_scalars.data(), numReqs, numDocs);
}

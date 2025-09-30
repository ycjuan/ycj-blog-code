#include "data.cuh"
#include "util.cuh"
#include <cuda_runtime.h>
#include <random>

namespace MatMatMulFromScratch
{

Data genData(int M, int N, int K)
{
    Data data;
    data.M = M;
    data.N = N;
    data.K = K;
    CHECK_CUDA(cudaMallocManaged(&data.d_A, M * K * sizeof(T)));
    CHECK_CUDA(cudaMallocManaged(&data.d_B, K * N * sizeof(T)));
    CHECK_CUDA(cudaMallocHost(&data.h_C, M * N * sizeof(float)));
    CHECK_CUDA(cudaMallocManaged(&data.d_C, M * N * sizeof(float)));

    std::default_random_engine generator;
    std::uniform_real_distribution<float> distribution(0.0, 1.0);
    for (int i = 0; i < data.M * data.K; i++)
    {
        data.d_A[i] = distribution(generator);
    }
    for (int i = 0; i < data.K * data.N; i++)
    {
        data.d_B[i] = distribution(generator);
    }

    return data;
}

void freeData(Data& data)
{
    if (data.d_A != nullptr)
    {
        cudaFree(data.d_A);
    }
    if (data.d_B != nullptr)
    {
        cudaFree(data.d_B);
    }
    if (data.h_C != nullptr)
    {
        cudaFreeHost(data.h_C);
    }
    if (data.d_C != nullptr)
    {
        cudaFree(data.d_C);
    }
}

} // namespace BatchScalability
#include <iostream>
#include <vector>
#include <random>
#include <cublas_v2.h>

#include "tasks.cuh"
#include "util.cuh"

void randomizeHostMatrix(__nv_bfloat16* h_matrix, uint64_t m, uint64_t n)
{
    std::default_random_engine generator;
    std::uniform_real_distribution<float> distribution(-1.0, 1.0);
    for (uint64_t i = 0; i < m * n; i++)
    {
        h_matrix[i] = __float2bfloat16(distribution(generator));
    }
}

void randomizeDeviceMatrix(__nv_bfloat16* d_matrix, uint64_t m, uint64_t n)
{
    std::vector<__nv_bfloat16> v_matrix(m * n);
    std::default_random_engine generator;
    std::uniform_real_distribution<float> distribution(-1.0, 1.0);
    for (auto& v : v_matrix)
    {
        v = __float2bfloat16(distribution(generator));
    }
    CHECK_CUDA(cudaMemcpy(d_matrix, v_matrix.data(), m * n * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));
}

BaseRunner::BaseRunner(uint64_t m, uint64_t n, uint64_t k)
    : m_m(m), m_n(n), m_k(k)
{
    std::cout << "MatMatMulRunner::MatMatMulRunner(" << m_m << ", " << m_n << ", " << m_k << ")" << std::endl;
    CHECK_CUDA(cudaMalloc(&m_d_A, m_m * m_k * sizeof(__nv_bfloat16)));
    CHECK_CUDA(cudaMalloc(&m_d_B, m_k * m_n * sizeof(__nv_bfloat16)));
    CHECK_CUDA(cudaMalloc(&m_d_C, m_m * m_n * sizeof(float)));

    randomizeDeviceMatrix(m_d_A, m_m, m_k);
    randomizeDeviceMatrix(m_d_B, m_k, m_n);
}

BaseRunner::~BaseRunner()
{
    std::cout << "MatMatMulRunner::~MatMatMulRunner()" << std::endl;
    cudaFree(m_d_A);
    cudaFree(m_d_B);
    cudaFree(m_d_C);
}

__global__ void matMatMulKernel(float* d_C, __nv_bfloat16* d_A, __nv_bfloat16* d_B, int m, int n, int k)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < m && col < n)
    {
        float sum = 0.0f;
        for (int i = 0; i < k; i++)
        {
            __nv_bfloat16 a = d_A[row * k + i];
            __nv_bfloat16 b = d_B[i * n + col];
            sum += (float)(a * b);
        }
        d_C[row * n + col] = sum;
    }
}

void CudaCoreMatMatMulRunner::run()
{
    std::cout << "CudaCoreMatMatMulRunner::run()" << std::endl;
    dim3 blockDim(16, 16);
    dim3 gridDim((m_m + blockDim.x - 1) / blockDim.x, (m_n + blockDim.y - 1) / blockDim.y);
    matMatMulKernel<<<gridDim, blockDim>>>(m_d_C, m_d_A, m_d_B, m_m, m_n, m_k);
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaGetLastError());
}

void TensorCoreMatMatMulRunner::run()
{
    std::cout << "TensorCoreMatMatMulRunner::run()" << std::endl;

    // Prepare CUBLAS handle
    cublasHandle_t handle;
    CHECK_CUBLAS(cublasCreate(&handle));

    const float alpha = 1.0f;
    const float beta = 0.0f;

    // Use cublasGemmEx for TensorCore-accelerated gemm
    // Note: assumes d_A_, d_B_, d_C_ are row-major, so to compute C = A x B:
    // Cublas expects column-major, so swap order & transpose. Or adjust as required.

    cublasGemmAlgo_t algo = CUBLAS_GEMM_DFALT_TENSOR_OP;
    cudaDataType_t computeType = CUDA_R_32F;

    // C = A x B, with A: (m_,k_), B: (k_,n_), C: (m_,n_)
    cublasStatus_t status = cublasGemmEx(
        handle,
        CUBLAS_OP_T, CUBLAS_OP_T,
        m_m, m_n, m_k,
        &alpha,
        m_d_A, CUDA_R_16BF, m_k,
        m_d_B, CUDA_R_16BF, m_n,
        &beta,
        m_d_C, CUDA_R_32F, m_m,
        computeType,
        algo
    );

    if (status != CUBLAS_STATUS_SUCCESS) 
    {
        std::cerr << "cublasGemmEx failed with status " << status << std::endl;
    }

    cublasDestroy(handle);
}

H2DMemcpyRunner::H2DMemcpyRunner(int m, int n, int k)
    : BaseRunner(m, n, k)
{
    std::cout << "H2DMemcpyRunner::H2DMemcpyRunner(" << m_m << ", " << m_n << ", " << m_k << ")" << std::endl;
    CHECK_CUDA(cudaMallocHost(&m_h_A, m_m * m_k * sizeof(__nv_bfloat16)));

    randomizeHostMatrix(m_h_A, m_m, m_k);
}

void H2DMemcpyRunner::run()
{
    std::cout << "H2DMemcpyRunner::run()" << std::endl;
    CHECK_CUDA(cudaMemcpy(m_d_A, m_h_A, m_m * m_k * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaGetLastError());
}
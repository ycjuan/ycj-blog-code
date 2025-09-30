#include "data.cuh"
#include "util.cuh"
#include <cublas_v2.h>

namespace MatMatMulFromScratch
{
#define cublasErrCheck(stat)                         \
    {                                                \
        cublasErrCheck_((stat), __FILE__, __LINE__); \
    }
void cublasErrCheck_(cublasStatus_t stat, const char* file, int line)
{
    if (stat != CUBLAS_STATUS_SUCCESS)
    {
        fprintf(stderr, "cuBLAS Error: %d %s %d\n", stat, file, line);
    }
}

void methodCublas(Data& data)
{
    cublasHandle_t cublasHandle;
    cublasCreate(&cublasHandle);

    float alpha = 1.0;
    float beta = 0.0;

    cublasOperation_t trana = (kAIsRowMajor) ? CUBLAS_OP_T : CUBLAS_OP_N;
    //cublasOperation_t tranb = (kBIsRowMajor) ? CUBLAS_OP_N : CUBLAS_OP_T;
    int lda = (kAIsRowMajor) ? data.K : data.M;
    //int ldb = (kBIsRowMajor) ? data.K : data.N;
    cudaDataType aType = (std::is_same<T, half>::value) ? CUDA_R_16F : CUDA_R_16BF;
    cudaDataType bType = (std::is_same<T, half>::value) ? CUDA_R_16F : CUDA_R_16BF;
    
    cublasErrCheck(cublasGemmEx(cublasHandle, trana, CUBLAS_OP_N, 
        data.M, data.N, data.K, 
        &alpha,
        data.d_A, aType, lda,
        data.d_B, bType, data.K,
        &beta, 
        data.d_C, CUDA_R_32F, data.M,
        CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));

    cublasDestroy(cublasHandle);
}

} // namespace MatMatMulFromScratch
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

    int MATRIX_M = data.M;
    int MATRIX_N = data.N;
    int MATRIX_K = data.K;

    T* a_fp16 = data.d_A;
    T* b_fp16 = data.d_B;
    float* c_cublas = data.d_C;

    //cublasOperation_t trana = (kAIsRowMajor) ? CUBLAS_OP_N : CUBLAS_OP_T;
    //cublasOperation_t tranb = (kBIsRowMajor) ? CUBLAS_OP_N : CUBLAS_OP_T;
    //int lda = (kAIsRowMajor) ? data.M : data.K;
    //int ldb = (kBIsRowMajor) ? data.K : data.N;
    //cudaDataType aType = (std::is_same<T, half>::value) ? CUDA_R_16F : CUDA_R_16BF;
    //cudaDataType bType = (std::is_same<T, half>::value) ? CUDA_R_16F : CUDA_R_16BF;
    
    cublasErrCheck(cublasGemmEx(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 
        MATRIX_M, MATRIX_N, MATRIX_K, 
        &alpha,
        a_fp16, CUDA_R_16F, MATRIX_M,
        b_fp16, CUDA_R_16F, MATRIX_K,
        &beta, 
        c_cublas, CUDA_R_32F, MATRIX_M,
        CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));

    cublasDestroy(cublasHandle);
}

} // namespace MatMatMulFromScratch
#ifndef METHOD_DOT_PROD_GPU_CUBLAS_CUH
#define METHOD_DOT_PROD_GPU_CUBLAS_CUH

#include <iostream>

#include "data.cuh"

using namespace std;

#define cublasErrCheck(stat) { cublasErrCheck_((stat), __FILE__, __LINE__); }
void cublasErrCheck_(cublasStatus_t stat, const char *file, int line) {
   if (stat != CUBLAS_STATUS_SUCCESS) {
      fprintf(stderr, "cuBLAS Error: %d %s %d\n", stat, file, line);
   }
}

template <typename T>
void matMulCublas(Data<T> data, int kNumTrials)
{
    cublasHandle_t cublasHandle;
    cublasCreate(&cublasHandle);

    float alpha = 1.0;
    float beta = 0.0;

    int MATRIX_M = data.numDocs;
    int MATRIX_N = data.numReqs;
    int MATRIX_K = data.embDim;
    T *a_fp16 = data.d_doc;
    T *b_fp16 = data.d_req;
    float *c_cublas = data.d_rst_cublas;

    cublasOperation_t trana = (data.docMemLayout == COL_MAJOR) ? CUBLAS_OP_N : CUBLAS_OP_T;
    cublasOperation_t tranb = (data.reqMemLayout == COL_MAJOR) ? CUBLAS_OP_T : CUBLAS_OP_N;
    int lda = (data.docMemLayout == COL_MAJOR) ? MATRIX_M : MATRIX_K;
    int ldb = (data.reqMemLayout == COL_MAJOR) ? MATRIX_N : MATRIX_K;
    cudaDataType aType = (is_same<T, half>::value) ? CUDA_R_16F : CUDA_R_16BF;
    cudaDataType bType = (is_same<T, half>::value) ? CUDA_R_16F : CUDA_R_16BF;

    CudaTimer timer;
    for (int t = -3; t < kNumTrials; t++)
    {
        if (t == 0)
            timer.tic();
        cublasErrCheck(cublasGemmEx(cublasHandle, trana, tranb,
                                    MATRIX_M, MATRIX_N, MATRIX_K,
                                    &alpha,
                                    a_fp16, aType, lda,
                                    b_fp16, bType, ldb,
                                    &beta,
                                    c_cublas, CUDA_R_32F, MATRIX_M,
                                    CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    }
    cout << "Cublas time: " << timer.tocMs() / kNumTrials << " ms" << endl;

    cublasDestroy(cublasHandle);
}

#endif
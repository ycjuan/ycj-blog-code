#include <string>
#include <stdexcept>
#include <iostream>
#include <random>
#include <sstream>
#include <cublas_v2.h>
#include <type_traits>
#include <omp.h>

#include "util.cuh"
#include "emb_data_struct.cuh"
#include "emb_op.cuh"

using namespace std;

#define CHECK_CUDA(func)                                                                                                           \
    {                                                                                                                              \
        cudaError_t status = (func);                                                                                               \
        if (status != cudaSuccess)                                                                                                 \
        {                                                                                                                          \
            string error = "CUDA API failed at line " + to_string(__LINE__) + " with error: " + cudaGetErrorString(status) + "\n"; \
            throw runtime_error(error);                                                                                            \
        }                                                                                                                          \
    }

#define CHECK_CUBLAS(func)                                                                                                  \
    {                                                                                                                       \
        cublasStatus_t status = (func);                                                                                     \
        if (status != CUBLAS_STATUS_SUCCESS)                                                                                \
        {                                                                                                                   \
            string error = "cuBLAS API failed at line " + to_string(__LINE__) + " with error: " + to_string(status) + "\n"; \
            throw runtime_error(error);                                                                                     \
        }                                                                                                                   \
    }

void embOpGpu(EmbData &data)
{
    CudaTimer timer;
    timer.tic();
    float alpha = 1.0;
    float beta = 0.0;

    int M = data.numDocs;
    int N = data.numReqs;
    int K = data.embDim;
    T_EMB *matA = data.d_doc;
    T_EMB *matB = data.d_req;
    float *matC = data.d_rst;

    cublasOperation_t tranA = (data.docMemLayout == COL_MAJOR) ? CUBLAS_OP_N : CUBLAS_OP_T;
    cublasOperation_t tranB = (data.reqMemLayout == COL_MAJOR) ? CUBLAS_OP_T : CUBLAS_OP_N;
    int ldA = (data.docMemLayout == COL_MAJOR) ? M : K;
    int ldB = (data.reqMemLayout == COL_MAJOR) ? N : K;
    // TODO: support float
    cudaDataType dataType = (is_same<T_EMB, float>::value) ? CUDA_R_32F : (is_same<T_EMB, half>::value) ? CUDA_R_16F : CUDA_R_16BF;

    CHECK_CUBLAS(cublasGemmEx(data.cublasHandle, tranA, tranB,
                              M, N, K,
                              &alpha,
                              matA, dataType, ldA,
                              matB, dataType, ldB,
                              &beta,
                              matC, CUDA_R_32F, M,
                              CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    CHECK_CUDA(cudaDeviceSynchronize());
    data.timeMsCuBlas = timer.tocMs();
}

void embOpCpu(EmbData &data)
{
    Timer timer;
    timer.tic();
    omp_set_num_threads(16);
    #pragma omp parallel for
    for (int i = 0; i < data.numDocs; i++)
    {
        for (int j = 0; j < data.numReqs; j++)
        {
            float sum = 0;
            for (int k = 0; k < data.embDim; k++)
            {
                T_EMB reqVal = data.h_req[getMemAddr(j, k, data.numReqs, data.embDim, data.reqMemLayout)];
                T_EMB docVal = data.h_doc[getMemAddr(i, k, data.numDocs, data.embDim, data.docMemLayout)];
                sum += (float)reqVal * (float)docVal;
            }
            data.h_rst[getMemAddr(i, j, data.numDocs, data.numReqs, data.rstMemLayout)] = (half)sum;
        }
    }
    data.timeMsCpu = timer.tocMs();
}
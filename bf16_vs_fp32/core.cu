#include <random>
#include <vector>
#include <stdexcept>
#include <iostream>
#include <cassert>
#include <algorithm>
#include <cuda_bf16.h>

#include "util.cuh"
#include "core.cuh"

using namespace std;

#define CHECK_CUDA(func)                                                                                                                     \
    {                                                                                                                                        \
        cudaError_t status = (func);                                                                                                         \
        if (status != cudaSuccess)                                                                                                           \
        {                                                                                                                                    \
            string error = "[main.cu] CUDA API failed at line " + to_string(__LINE__) + " with error: " + cudaGetErrorString(status) + "\n"; \
            throw runtime_error(error);                                                                                                      \
        }                                                                                                                                    \
    }

template <typename T_EMB, typename T_ACC>
__global__ void kernel_fp32_bf16(T_EMB *d_docEmb, T_EMB *d_reqEmb, Doc *d_doc, int numAllDocs, int numActiveDocs, int embDim)
{
    int d = blockIdx.x * blockDim.x + threadIdx.x;
    if (d < numDocs)
    {
        Doc &doc = d_doc[d];
        T_ACC acc = 0;
        int i = doc.docIdx;
        for (int j = 0; j < embDim; j++)
        {
            T_EMB docVal = d_docEmb[getMemAddr(i, j, numAllDocs, embDim)];
            T_EMB reqVal = d_docEmb[getMemAddr(i, j, numAllDocs, embDim)];
            acc += (TACC)(docVal * reqVal);
        }
        doc.score = (float)acc;
    }
}

template <typename T_EMB, typename T_ACC>
float score_fp32_bf16(T_EMB *d_docEmb, T_EMB *d_reqEmb, Doc *d_doc, int numAllDocs, int numActiveDocs, int embDim)
{
    CudaTimer timer;
    timer.tic();
    int gridSize = (int)ceil((double)numActiveDocs / kBlockSize);
    kernel_fp32_bf16<T_EMB, T_ACC><<<gridSize, kBlockSize>>>(d_docEmb, d_reqEmb, d_doc, numAllDocs, numActiveDocs, embDim);
    cudaDeviceSynchronize();
    CHECK_CUDA(cudaGetLastError());
    return timer.tocMs();
}
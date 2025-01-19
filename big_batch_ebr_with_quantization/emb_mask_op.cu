#include <string>
#include <stdexcept>
#include <iostream>
#include <random>
#include <sstream>
#include <cublas_v2.h>
#include <type_traits>
#include <omp.h>

#include "util.cuh"
#include "emb.cuh"

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

__global__ void embMaskOpKernel(EmbData data)
{
    int maskIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (maskIdx >= data.maskSize)
        return;

    Pair &mask = data.d_mask[maskIdx];
    float score = 0;
    for (int k = 0; k < data.embDim; k++)
    {
        T_EMB reqVal = data.d_req[getMemAddr(mask.reqIdx, k, data.numReqs, data.embDim, data.reqMemLayout)];
        T_EMB docVal = data.d_doc[getMemAddr(mask.docIdx, k, data.numDocs, data.embDim, data.docMemLayout)];
        T_EMB score1 = reqVal * docVal;
        score += (float)score1;
    }
    mask.score = score;
}

void embMaskOpGpu(EmbData &data)
{
    CudaTimer timer;
    timer.tic();
    int blockSize = 256;
    int numBlocks = (data.maskSize + blockSize - 1) / blockSize;
    embMaskOpKernel<<<numBlocks, blockSize>>>(data);
    data.timeMsCuBlas = timer.tocMs();
}

void embMaskOpCpu(EmbData &data)
{
    Timer timer;
    timer.tic();
    omp_set_num_threads(16);
#pragma omp parallel for
    for (size_t maskIdx = 0; maskIdx < data.maskSize; maskIdx++)
    {
        Pair &mask = data.h_mask[maskIdx];
        float score = 0;
        for (int k = 0; k < data.embDim; k++)
        {
            T_EMB reqVal = data.h_req[getMemAddr(mask.reqIdx, k, data.numReqs, data.embDim, data.reqMemLayout)];
            T_EMB docVal = data.h_doc[getMemAddr(mask.docIdx, k, data.numDocs, data.embDim, data.docMemLayout)];
            score += (float)reqVal * (float)docVal;
        }
        mask.score = score;
    }
    data.timeMsCpu = timer.tocMs();
}
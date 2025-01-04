#include <iostream>

#include "util.cuh"
#include "methods.cuh"

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

__device__ void func1(Param param, size_t wid)
{
    param.d_count[wid] = wid;
    for (int i = 0; i < param.numCountInc; i++)
    {
        param.d_count[wid] += 1;
    }
}

__global__ void setupAKernel(Param param)
{
    size_t wid = (size_t)blockIdx.x * blockDim.x + threadIdx.x;

    if (wid < param.dataSize)
    {
        func1(param, wid);
    }
}

void runSetupA(Param param)
{
    int blockSize = 256;
    int numBlocks = (param.dataSize + blockSize - 1) / blockSize;

    CudaTimer timer;
    timer.tic();
    for (int i = 0; i < param.numTrials; i++)
    {
        setupAKernel<<<numBlocks, blockSize>>>(param);
        CHECK_CUDA(cudaDeviceSynchronize());
        CHECK_CUDA(cudaGetLastError());
    }
    cout << "setupAKernel time: " << timer.tocMs() << " ms" << endl;
}
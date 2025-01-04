#include <iostream>

#include "util.cuh"
#include "methods.cuh"
#include "setupB.cuh"

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

__global__ void setupBKernel(Param param, FuncRunnerB funcRunnerB)
{
    size_t wid = (size_t)blockIdx.x * blockDim.x + threadIdx.x;

    if (wid < param.dataSize)
    {
        funcRunnerB.runFunc(param, wid);
    }
}

void runSetupB(Param param)
{
    int blockSize = 256;
    int numBlocks = (param.dataSize + blockSize - 1) / blockSize;

    FuncRunnerB funcRunnerB;

    CudaTimer timer;
    timer.tic();
    for (int i = -3; i < param.numTrials; i++)
    {
        setupBKernel<<<numBlocks, blockSize>>>(param, funcRunnerB);
        CHECK_CUDA(cudaDeviceSynchronize());
        CHECK_CUDA(cudaGetLastError());
    }
    cout << "setupBKernel time: " << timer.tocMs() << " ms" << endl;
}
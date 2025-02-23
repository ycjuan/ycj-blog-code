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

__global__ void baselineKernel(Param param)
{
    size_t taskId = (size_t)blockIdx.x * blockDim.x + threadIdx.x;

    if (taskId < param.dataSize)
    {
        param.d_count[taskId] = taskId;
        for (int i = 0; i < param.numCountInc; i++)
        {
            if (taskId == 1)
                continue;
            else if (taskId == 2)
                continue;
            else if (taskId == 3)
                continue;
            else if (taskId == 4)
                continue;
            else if (taskId == 5)
                continue;
            else if (taskId == 6)
                continue;
            else if (taskId == 7)
                continue;
            param.d_count[taskId]++;
        }
    }
}

void runSetupBaseline(Param param)
{
    int blockSize = 256;
    int numBlocks = (param.dataSize + blockSize - 1) / blockSize;

    CudaTimer timer;
    timer.tic();
    for (int i = -3; i < param.numTrials; i++)
    {
        baselineKernel<<<numBlocks, blockSize>>>(param);
        CHECK_CUDA(cudaDeviceSynchronize());
        CHECK_CUDA(cudaGetLastError());
    }
    cout << "baselineKernel time: " << timer.tocMs() << " ms" << endl;
}
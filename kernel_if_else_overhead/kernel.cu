#include <iostream>

#include "util.cuh"
#include "common.cuh"

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

__global__ void method0Kernel(Param param)
{
    size_t taskId = (size_t)blockIdx.x * blockDim.x + threadIdx.x;

    if (taskId < param.dataSize)
    {
        param.d_count[taskId] = taskId;
        for (int i = 0; i < param.numCountInc; i++)
        {
            param.d_count[taskId] += 1;
        }
    }
}

__global__ void method1Kernel(Param param)
{
    size_t taskId = (size_t)blockIdx.x * blockDim.x + threadIdx.x;

    if (taskId < param.dataSize)
    {
        param.d_count[taskId] = taskId;
        for (int i = 0; i < param.numCountInc; i++)
        {
            if (taskId == 1)
                continue;
            param.d_count[taskId] += 1;
        }
    }
}

__global__ void method2Kernel(Param param)
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
            param.d_count[taskId] += 1;
        }
    }
}

__global__ void method3Kernel(Param param)
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
            param.d_count[taskId] += 1;
        }
    }
}

void runExp(Param param)
{
    int blockSize = 256;
    int numBlocks = (param.dataSize + blockSize - 1) / blockSize;

    CudaTimer timer;
    timer.tic();
    for (int i = -3; i < param.numTrials; i++)
    {
        if (param.method == METHOD0)
            method0Kernel<<<numBlocks, blockSize>>>(param);
        else if (param.method == METHOD1)
            method1Kernel<<<numBlocks, blockSize>>>(param);
        else if (param.method == METHOD2)
            method2Kernel<<<numBlocks, blockSize>>>(param);
        else if (param.method == METHOD3)
            method3Kernel<<<numBlocks, blockSize>>>(param);
        else
            throw runtime_error("Invalid method");
        CHECK_CUDA(cudaDeviceSynchronize());
        CHECK_CUDA(cudaGetLastError());
    }
    cout << "Method " << param.method << " time: " << timer.tocMs() << " ms" << endl;
}
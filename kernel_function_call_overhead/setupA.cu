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

__device__ void func0(Param param, size_t taskId)
{
    param.d_count[taskId] = taskId;
    for (int i = 0; i < param.numCountInc; i++)
    {
        param.d_count[taskId] += 1;
    }
}

__device__ void func1(Param param, size_t taskId)
{
    func0(param, taskId);
}

__device__ void func2(Param param, size_t taskId)
{
    func1(param, taskId);
}

__device__ void func3(Param param, size_t taskId)
{
    func2(param, taskId);
}

__device__ void func4(Param param, size_t taskId)
{
    func3(param, taskId);
}

__device__ void func5(Param param, size_t taskId)
{
    func4(param, taskId);
}

__device__ void func6(Param param, size_t taskId)
{
    func5(param, taskId);
}

__device__ void func7(Param param, size_t taskId)
{
    func6(param, taskId);
}

__global__ void setupAKernel(Param param)
{
    size_t taskId = (size_t)blockIdx.x * blockDim.x + threadIdx.x;

    if (taskId < param.dataSize)
    {
        func7(param, taskId);
    }
}

void runSetupA(Param param)
{
    int blockSize = 256;
    int numBlocks = (param.dataSize + blockSize - 1) / blockSize;

    CudaTimer timer;
    timer.tic();
    for (int i = -3; i < param.numTrials; i++)
    {
        setupAKernel<<<numBlocks, blockSize>>>(param);
        CHECK_CUDA(cudaDeviceSynchronize());
        CHECK_CUDA(cudaGetLastError());
    }
    cout << "setupAKernel time: " << timer.tocMs() << " ms" << endl;
}
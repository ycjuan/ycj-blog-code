#include <iostream>

#include "util.cuh"

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

__global__ void baselineKernel(long *d_count, int dataSize, int numCountInc)
{
    size_t wid = (size_t)blockIdx.x * blockDim.x + threadIdx.x;

    if (wid < dataSize)
    {
        d_count[wid] = wid;
        for (int i = 0; i < numCountInc; i++)
        {
            d_count[wid] += 1;
        }
    }
}

void runSetupBaseline(long *d_count, long size, int numCountInc, int numTrials)
{
    int blockSize = 256;
    int numBlocks = (size + blockSize - 1) / blockSize;

    CudaTimer timer;
    timer.tic();
    for (int i = 0; i < numTrials; i++)
    {
        baselineKernel<<<numBlocks, blockSize>>>(d_count, size, numCountInc);
        CHECK_CUDA(cudaDeviceSynchronize());
        CHECK_CUDA(cudaGetLastError());
    }
    cout << "baselineKernel time: " << timer.tocMs() << " ms" << endl;
}
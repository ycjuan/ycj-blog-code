#include "util.cuh"
#include <bits/types/struct_sched_param.h>
#include <iostream>
#include <vector>
#include <cuda_runtime.h>

constexpr int kNumReqs = 1024;
constexpr int kNumDocs = 1000;
constexpr int kNumRepeats = 1000;
constexpr int kNumTrials = 100;
constexpr int kBlockSize = 1024;

void printConfig()
{
    std::cout << "kNumReqs: " << kNumReqs << std::endl;
    std::cout << "kNumDocs: " << kNumDocs << std::endl;
    std::cout << "kNumRepeats: " << kNumRepeats << std::endl;
    std::cout << "kNumTrials: " << kNumTrials << std::endl;
    std::cout << "kBlockSize: " << kBlockSize << std::endl;
}

__global__ void dummyKernel(long *d_rst, int reqIdx)
{
    int docIdx = threadIdx.x + blockIdx.x * blockDim.x;
    if (docIdx < kNumDocs)
    {
        int rst = docIdx + reqIdx * kNumDocs;
        for (int i = 0; i < kNumRepeats; i++)
        {
            // Some pseudo-random computation to avoid compiler optimization
            rst = (rst * 1103515245 + 12345) & 0x7fffffff;\
        }
        d_rst[docIdx + reqIdx * kNumDocs] = rst;
    }
}

void runReqByReq()
{
    // ----------------
    // Preparation
    long *d_rst;
    CHECK_CUDA(cudaMalloc(&d_rst, kNumReqs * kNumDocs * sizeof(long)));

    // ----------------
    // Run experiment
    Timer timer;
    for (int t = -3; t < kNumTrials; t++)
    {
        if (t == 0)
            timer.tic();
        for (int reqIdx = 0; reqIdx < kNumReqs; reqIdx++)
        {
            int gridSize = (int)ceil((double)(kNumDocs + 1) / kBlockSize);
            dummyKernel<<<gridSize, kBlockSize>>>(d_rst, reqIdx);
            CHECK_CUDA(cudaDeviceSynchronize());
            CHECK_CUDA(cudaGetLastError());
        }
    }
    float timeMs = timer.tocMs() / kNumTrials;
    std::cout << "Time taken for req by req: " << timeMs << " ms" << std::endl;

    // ----------------
    // Cleanup
    CHECK_CUDA(cudaFree(d_rst));
}

void runParallelWithCudaStream()
{
    // ----------------
    // Preparation
    long *d_rst;
    CHECK_CUDA(cudaMalloc(&d_rst, kNumReqs * kNumDocs * sizeof(long)));
    std::vector<cudaStream_t> streams;
    for (int i = 0; i < kNumReqs; i++)
    {
        cudaStream_t stream;
        CHECK_CUDA(cudaStreamCreate(&stream));
        streams.push_back(stream);
    }

    // ----------------
    // Run experiment
    Timer timer;

    for (int t = -3; t < kNumTrials; t++)
    {
        if (t == 0)
            timer.tic();
        for (int reqIdx = 0; reqIdx < kNumReqs; reqIdx++)
        {   
            int gridSize = (int)ceil((double)(kNumDocs + 1) / kBlockSize);
            dummyKernel<<<gridSize, kBlockSize, 0, streams[reqIdx]>>>(d_rst, reqIdx);
        }
        for (int i = 0; i < kNumReqs; i++)
        {
            CHECK_CUDA(cudaStreamSynchronize(streams[i]));
        }
    }
    float timeMs = timer.tocMs() / kNumTrials;
    std::cout << "Time taken for parallel with cuda stream: " << timeMs << " ms" << std::endl;

    // ----------------
    // Cleanup
    CHECK_CUDA(cudaFree(d_rst));
    for (int i = 0; i < kNumReqs; i++)
    {
        CHECK_CUDA(cudaStreamDestroy(streams[i]));
    }
}

int main()
{
    printDeviceInfo();
    
    printConfig();
    
    runReqByReq();

    runParallelWithCudaStream();

    return 0;
}
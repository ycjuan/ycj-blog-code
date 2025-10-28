#include "util.cuh"
#include <bits/types/struct_sched_param.h>
#include <cassert>
#include <cuda_runtime.h>
#include <iostream>
#include <vector>

struct Config
{
    int numReqs;
    int numDocs;
    int numRepeats;
    int numTrials;

    const int kBlockSize = 1024;

    void print()
    {
        std::cout << "numReqs: " << numReqs << std::endl;
        std::cout << "numDocs: " << numDocs << std::endl;
        std::cout << "numRepeats: " << numRepeats << std::endl;
        std::cout << "numTrials: " << numTrials << std::endl;
    }
};

__global__ void dummyKernel(long* d_rst, int reqIdx, int numDocs, int numRepeats)
{
    int docIdx = threadIdx.x + blockIdx.x * blockDim.x;
    if (docIdx < numDocs)
    {
        int rst = docIdx + reqIdx * numDocs;
        for (int i = 0; i < numRepeats; i++)
        {
            // Some pseudo-random computation to avoid compiler optimization
            rst = (rst * 1103515245 + 12345) & 0x7fffffff;
        }
        d_rst[docIdx + reqIdx * numDocs] = rst;
    }
}

void runReqByReq(Config config)
{
    // ----------------
    // Preparation
    long* d_rst;
    CHECK_CUDA(cudaMalloc(&d_rst, config.numReqs * config.numDocs * sizeof(long)));

    // ----------------
    // Run experiment
    Timer timer;
    for (int t = -3; t < config.numTrials; t++)
    {
        if (t == 0)
            timer.tic();
        for (int reqIdx = 0; reqIdx < config.numReqs; reqIdx++)
        {
            int gridSize = (int)ceil((double)(config.numDocs + 1) / config.kBlockSize);
            dummyKernel<<<gridSize, config.kBlockSize>>>(d_rst, reqIdx, config.numDocs, config.numRepeats);
        }
    }
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaGetLastError());
    float timeMs = timer.tocMs() / config.numTrials;
    std::cout << "Time taken for req by req: " << timeMs << " ms" << std::endl;

    // ----------------
    // Cleanup
    CHECK_CUDA(cudaFree(d_rst));
}

void runParallelWithCudaStream(Config config, int numCudaStreams)
{
    // ----------------
    // Preparation
    long* d_rst;
    CHECK_CUDA(cudaMalloc(&d_rst, config.numReqs * config.numDocs * sizeof(long)));
    std::vector<cudaStream_t> streams;
    for (int i = 0; i < numCudaStreams; i++)
    {
        cudaStream_t stream;
        CHECK_CUDA(cudaStreamCreate(&stream));
        streams.push_back(stream);
    }

    // ----------------
    // Run experiment
    Timer timer;

    for (int t = -3; t < config.numTrials; t++)
    {
        if (t == 0)
            timer.tic();
        for (int reqIdx = 0; reqIdx < config.numReqs; reqIdx++)
        {
            int gridSize = (int)ceil((double)(config.numDocs + 1) / config.kBlockSize);
            dummyKernel<<<gridSize, config.kBlockSize, 0, streams[reqIdx % numCudaStreams]>>>(
                d_rst, reqIdx, config.numDocs, config.numRepeats);
        }
        for (int i = 0; i < numCudaStreams; i++)
        {
            CHECK_CUDA(cudaStreamSynchronize(streams[i]));
        }
    }
    float timeMs = timer.tocMs() / config.numTrials;
    std::cout << "Time taken with " << numCudaStreams << " cuda streams: " << timeMs << " ms" << std::endl;

    // ----------------
    // Cleanup
    CHECK_CUDA(cudaFree(d_rst));
    for (int i = 0; i < numCudaStreams; i++)
    {
        CHECK_CUDA(cudaStreamDestroy(streams[i]));
    }
}

int main()
{
    printDeviceInfo();

    {
        Config config;
        config.numReqs = 32;
        config.numDocs = 1000000;
        config.numRepeats = 10;
        config.numTrials = 100;

        config.print();

        runReqByReq(config);

        for (int numCudaStreams = 1; numCudaStreams <= config.numReqs; numCudaStreams *= 2)
        {
            assert(config.numReqs % numCudaStreams == 0);
            runParallelWithCudaStream(config, numCudaStreams);
        }
    }

    std::cout << "\n--------------------------------" << std::endl;

    {
        Config config;
        config.numReqs = 16384;
        config.numDocs = 1000;
        config.numRepeats = 100;
        config.numTrials = 100;

        config.print();

        runReqByReq(config);

        for (int numCudaStreams = 1; numCudaStreams <= config.numReqs; numCudaStreams *= 2)
        {
            assert(config.numReqs % numCudaStreams == 0);
            runParallelWithCudaStream(config, numCudaStreams);
        }
    }

    std::cout << "\n--------------------------------" << std::endl;

    {
        Config config;
        config.numReqs = 16384;
        config.numDocs = 1000;
        config.numRepeats = 1000;
        config.numTrials = 100;

        config.print();

        runReqByReq(config);

        for (int numCudaStreams = 1; numCudaStreams <= config.numReqs; numCudaStreams *= 2)
        {
            assert(config.numReqs % numCudaStreams == 0);
            runParallelWithCudaStream(config, numCudaStreams);
        }
    }

    return 0;
}
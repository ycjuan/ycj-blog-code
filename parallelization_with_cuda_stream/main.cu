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

std::vector<std::vector<long>> getCpuReference(Config config)
{
    std::vector<std::vector<long>> rst2D(config.numReqs, std::vector<long>(config.numDocs));
    #pragma omp parallel for
    for (int reqIdx = 0; reqIdx < config.numReqs; reqIdx++)
    {
        for (int docIdx = 0; docIdx < config.numDocs; docIdx++)
        {
            long rst = docIdx + reqIdx * config.numDocs;
            for (int i = 0; i < config.numRepeats; i++)
            {
                rst = (rst * 1103515245 + 12345) & 0x7fffffff;
                rst = (rst >> 3) ^ (rst << 7);
                rst = (rst * 16807) % 2147483647;
            }
            rst2D[reqIdx][docIdx] = rst;
        }
    }
    return rst2D;
}

__global__ void dummyKernel(long* d_rst, int reqIdx, int numDocs, int numRepeats)
{
    int docIdx = threadIdx.x + blockIdx.x * blockDim.x;
    if (docIdx < numDocs)
    {
        long rst = docIdx + reqIdx * numDocs;
        for (int i = 0; i < numRepeats; i++)
        {
            // Some pseudo-random computation to avoid compiler optimization
            rst = (rst * 1103515245 + 12345) & 0x7fffffff;
            rst = (rst >> 3) ^ (rst << 7);
            rst = (rst * 16807) % 2147483647;            
        }
        d_rst[docIdx + reqIdx * numDocs] = rst;
    }
}

void runReqByReq(Config config, const std::vector<std::vector<long>>& cpuReference)
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
    // Check correctness
    long *h_rst;
    CHECK_CUDA(cudaMallocHost(&h_rst, config.numReqs * config.numDocs * sizeof(long)));
    CHECK_CUDA(cudaMemcpy(h_rst, d_rst, config.numReqs * config.numDocs * sizeof(long), cudaMemcpyDeviceToHost));
    for (int reqIdx = 0; reqIdx < config.numReqs; reqIdx++)
    {
        for (int docIdx = 0; docIdx < config.numDocs; docIdx++)
        {
            assert(h_rst[reqIdx * config.numDocs + docIdx] == cpuReference[reqIdx][docIdx]);
        }
    }
    std::cout << "All results are correct ^____^" << std::endl;

    // ----------------
    // Cleanup
    CHECK_CUDA(cudaFree(d_rst));
    CHECK_CUDA(cudaFreeHost(h_rst));
}

void runParallelWithCudaStream(Config config, int numCudaStreams, const std::vector<std::vector<long>>& cpuReference)
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
    // Check correctness
    long *h_rst;
    CHECK_CUDA(cudaMallocHost(&h_rst, config.numReqs * config.numDocs * sizeof(long)));
    CHECK_CUDA(cudaMemcpy(h_rst, d_rst, config.numReqs * config.numDocs * sizeof(long), cudaMemcpyDeviceToHost));
    for (int reqIdx = 0; reqIdx < config.numReqs; reqIdx++)
    {
        for (int docIdx = 0; docIdx < config.numDocs; docIdx++)
        {
            assert(h_rst[reqIdx * config.numDocs + docIdx] == cpuReference[reqIdx][docIdx]);
        }
    }
    std::cout << "All results are correct ^____^" << std::endl;

    // ----------------
    // Cleanup
    CHECK_CUDA(cudaFree(d_rst));
    CHECK_CUDA(cudaFreeHost(h_rst));
}

void runOneConfig(Config config)
{
    const auto cpuReference = getCpuReference(config);

    runReqByReq(config, cpuReference);

    for (int numCudaStreams = 1; numCudaStreams <= config.numReqs; numCudaStreams *= 2)
    {
        assert(config.numReqs % numCudaStreams == 0);
        runParallelWithCudaStream(config, numCudaStreams, cpuReference);
    }
}

int main()
{
    printDeviceInfo();
    
    {
        Config config;
        config.numReqs = 32;
        config.numDocs = 1000000;
        config.numRepeats = 1000;
        config.numTrials = 100;
        config.print();

        runOneConfig(config);
    }

    std::cout << "\n--------------------------------" << std::endl;

    {
        Config config;
        config.numReqs = 16384;
        config.numDocs = 1000;
        config.numRepeats = 100;
        config.numTrials = 100;
        config.print();

        runOneConfig(config);
    }

    std::cout << "\n--------------------------------" << std::endl;

    {
        Config config;
        config.numReqs = 16384;
        config.numDocs = 1000;
        config.numRepeats = 1000;
        config.numTrials = 100;
        config.print();

        runOneConfig(config);
    }
}
#include <vector>
#include <cmath>
#include <functional>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <cassert>
#include <numeric>
#include <thrust/copy.h>
#include <thrust/execution_policy.h>
#include <thrust/sort.h>
#include <string>
#include <omp.h>

#include "topk.cuh"
#include "common.cuh"
#include "util.cuh"

// Note: This is experimental code, so corner cases such as "numToRetrieve > numDocs" are not handled

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

void TopkSampling::malloc()
{
    size_t allocateInBytes;
    size_t totalAllocateInBytes = 0;

    allocateInBytes = kNumSamplesPerReq * kMaxNumReqs * sizeof(float);
    CHECK_CUDA(cudaMalloc(&d_scoreSample, allocateInBytes));
    totalAllocateInBytes += allocateInBytes;
    cout << "allocated " << allocateInBytes / 1024.0 / 1024.0 / 1024.0 << " GiB for d_scoreSample" << endl;

    allocateInBytes = kMaxNumReqs * sizeof(float);
    CHECK_CUDA(cudaMallocManaged(&dm_scoreThreshold, allocateInBytes));
    totalAllocateInBytes += allocateInBytes;
    cout << "allocated " << allocateInBytes / 1024.0 / 1024.0 / 1024.0 << " GiB for dm_scoreThreshold" << endl;

    allocateInBytes = kMaxNumReqs * kMaxEligiblePairsPerReq * sizeof(Pair);
    CHECK_CUDA(cudaMalloc(&d_eligiblePairs, allocateInBytes));
    totalAllocateInBytes += allocateInBytes;
    cout << "allocated " << allocateInBytes / 1024.0 / 1024.0 / 1024.0 << " GiB for d_eligiblePairs" << endl;

    allocateInBytes = kMaxNumReqs * sizeof(int);
    CHECK_CUDA(cudaMallocManaged(&dm_copyCount, allocateInBytes));
    totalAllocateInBytes += allocateInBytes;
    cout << "allocated " << allocateInBytes / 1024.0 / 1024.0 / 1024.0 << " GiB for dm_copyCount" << endl;

    allocateInBytes = kMaxEligiblePairsPerReq * sizeof(Pair) + 10000000;
    thrustAllocator.malloc(allocateInBytes);
    totalAllocateInBytes += allocateInBytes;
    cout << "allocated " << allocateInBytes / 1024.0 / 1024.0 / 1024.0 << " GiB for thrustAllocator" << endl;

    cout << "total allocated " << totalAllocateInBytes / 1024.0 / 1024.0 / 1024.0 << " GiB" << endl;
}

void TopkSampling::free()
{
    CHECK_CUDA(cudaFree(d_scoreSample));
    CHECK_CUDA(cudaFree(dm_scoreThreshold));
    CHECK_CUDA(cudaFree(d_eligiblePairs));
    CHECK_CUDA(cudaFree(dm_copyCount));
    thrustAllocator.free();
}

void TopkSampling::retrieveTopk(TopkParam &param)
{
    CudaTimer timerTotal;
    CudaTimer timerApprox;
    timerTotal.tic();
    timerApprox.tic();

    // Step1 - Sample
    sample(param);

    // Step2 - Find threshold
    findThreshold(param);

    // Step3 - Copy eligible
    copyEligible(param);
    param.gpuApproxTimeMs = timerApprox.tocMs();

    // Step4 - retreiveExact
    retrieveExact(param);

    param.gpuTotalTimeMs = timerTotal.tocMs();
}

__global__ void sampleKernelNonRandom(TopkParam topkParam, float *d_scoreSample, size_t sampleSizePerReq)
{
    int wid = threadIdx.x + blockIdx.x * blockDim.x;

    if (wid < topkParam.numReqs * sampleSizePerReq)
    {
        int reqIdx = wid % topkParam.numReqs;
        int docIdx = wid / topkParam.numReqs;
        d_scoreSample[getMemAddr(reqIdx, docIdx, sampleSizePerReq)] = topkParam.dm_score[getMemAddr(reqIdx, docIdx, topkParam.numDocs)];
    }
}

void TopkSampling::sample(TopkParam &param)
{
    CudaTimer timer;
    timer.tic();

    int blockSize = 256;
    int gridSize = (param.numReqs * kNumSamplesPerReq + blockSize - 1) / blockSize;
    sampleKernelNonRandom<<<gridSize, blockSize>>>(param, d_scoreSample, kNumSamplesPerReq);
    CHECK_CUDA(cudaDeviceSynchronize());

    param.gpuSampleTimeMs = timer.tocMs();
}

__global__ void updateThreshold(float *d_scoreSample, float *dm_scoreThreshold, int numReqs, int thIdx, size_t kNumSamplesPerReq)
{
    int reqIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (reqIdx < numReqs)
    {
        dm_scoreThreshold[reqIdx] = d_scoreSample[reqIdx * kNumSamplesPerReq + thIdx];
    }
}

void TopkSampling::findThreshold(TopkParam &param)
{
    CudaTimer timer;
    timer.tic();

    int thIdx = ceil((double)param.numToRetrieve / param.numDocs * kNumSamplesPerReq * 4);
    //omp_set_num_threads(4);
    //#pragma omp parallel for
    // mutlithreading does not help much here
    for (size_t reqIdx = 0; reqIdx < param.numReqs; reqIdx++)
    {
        thrust::sort(thrust::cuda::par(thrustAllocator),
                     d_scoreSample + reqIdx       * kNumSamplesPerReq,
                     d_scoreSample + (reqIdx + 1) * kNumSamplesPerReq,
                     thrust::greater<float>());
    }

    int blockSize = 256;
    int gridSize = (param.numReqs + blockSize - 1) / blockSize;
    updateThreshold<<<gridSize, blockSize>>>(d_scoreSample, dm_scoreThreshold, param.numReqs, thIdx, kNumSamplesPerReq);

    param.gpuFindThresholdTimeMs = timer.tocMs();
}

__global__ void copyEligibleKernel(float *dm_score,
                                   float *dm_scoreThreshold,
                                   Pair *d_eligiblePairs,
                                   int *dm_copyCount,
                                   int numReqs,
                                   int numDocs,
                                   size_t kMaxEligiblePairsPerReq)
{
    size_t wid = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (wid < numReqs * numDocs)
    {
        int reqIdx = wid % numReqs;
        int docIdx = wid / numReqs;
        size_t memAddr = getMemAddr(reqIdx, docIdx, numDocs);
        float score = dm_score[memAddr];
        float threshold = dm_scoreThreshold[reqIdx];
        if (score >= threshold)
        {
            int count = atomicAdd(dm_copyCount + reqIdx, 1);
            if (count < kMaxEligiblePairsPerReq)
            {
                Pair pair;
                pair.reqIdx = reqIdx;
                pair.docIdx = docIdx;
                pair.score = score;
                d_eligiblePairs[reqIdx * kMaxEligiblePairsPerReq + count] = pair;
            }
        }
    }
}

void TopkSampling::copyEligible(TopkParam &param)
{
    /*
    for (size_t reqIdx = 0; reqIdx < param.numReqs; reqIdx++)
    {
        thrust::copy_if(thrust::device,
                        param.dm_score + reqIdx * param.numDocs,
                        param.dm_score + (reqIdx + 1) * param.numDocs,
                        param.dm_rst + reqIdx * param.numToRetrieve,
                        [thIdx = dm_scoreThreshold[reqIdx]] __device__(float score) mutable {
                            return score >= thIdx;
                        });
    }
    */

    CudaTimer timer;
    timer.tic();

    CHECK_CUDA(cudaMemset(dm_copyCount, 0, param.numReqs * sizeof(int)));
    int blockSize = 256;
    int gridSize = (param.numReqs * param.numDocs + blockSize - 1) / blockSize;
    copyEligibleKernel<<<gridSize, blockSize>>>(param.dm_score,
                                                dm_scoreThreshold,
                                                d_eligiblePairs,
                                                dm_copyCount,
                                                param.numReqs,
                                                param.numDocs,
                                                kMaxEligiblePairsPerReq);

    CHECK_CUDA(cudaDeviceSynchronize());

    param.gpuCopyEligibleTimeMs = timer.tocMs();
}

void TopkSampling::retrieveExact(TopkParam &param)
{
    CudaTimer timer;
    timer.tic();
    for (size_t reqIdx = 0; reqIdx < param.numReqs; reqIdx++)
    {
        Pair *dm_eligiblePairsStart = d_eligiblePairs + reqIdx * kMaxEligiblePairsPerReq;
        Pair *dm_eligiblePairsEnd = dm_eligiblePairsStart + dm_copyCount[reqIdx];
        thrust::stable_sort(thrust::cuda::par(thrustAllocator),
                            dm_eligiblePairsStart,
                            dm_eligiblePairsEnd,
                            ScorePredicator());
        thrust::copy(thrust::cuda::par(thrustAllocator),
                     dm_eligiblePairsStart,
                     dm_eligiblePairsStart + param.numToRetrieve,
                     param.dm_rstGpu + reqIdx * param.numToRetrieve);
    }
    param.gpuRetreiveExactTimeMs = timer.tocMs();
}
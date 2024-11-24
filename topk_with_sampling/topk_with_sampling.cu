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
            string error = "[topk_with_bucket_sort.cu] CUDA API failed at line " + to_string(__LINE__) + " with error: " + cudaGetErrorString(status) + "\n"; \
            throw runtime_error(error);                                                                                            \
        }                                                                                                                          \
    }

void TopkSampling::malloc()
{
    CHECK_CUDA(cudaMallocManaged(&dm_scoreSample, kNumSamplesPerReq * kMaxNumReqs * sizeof(float)));
    CHECK_CUDA(cudaMallocManaged(&dm_scoreThreshold, kMaxNumReqs * sizeof(float)));
    CHECK_CUDA(cudaMallocManaged(&dm_eligiblePairs, kMaxNumReqs * kMaxEligiblePairsPerDoc * sizeof(Pair)));
    CHECK_CUDA(cudaMallocManaged(&dm_copyCount, kMaxNumReqs * sizeof(int)));
}

void TopkSampling::free()
{
    CHECK_CUDA(cudaFree(dm_scoreSample));
    CHECK_CUDA(cudaFree(dm_scoreThreshold));
    CHECK_CUDA(cudaFree(dm_eligiblePairs));
    CHECK_CUDA(cudaFree(dm_copyCount));
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
    size_t numCopied = 0;
    copyEligible(param);
    param.gpuApproxTimeMs = timerApprox.tocMs();

    // Step4 - retreiveExact
    retrieveExact(param);

    param.gpuTotalTimeMs = timerTotal.tocMs();
}

__global__ void sampleKernelNonRandom(TopkParam &topkParam, float *dm_scoreSample, size_t sampleSizePerReq)
{
    int wid = threadIdx.x + blockIdx.x * blockDim.x;

    if (wid < topkParam.numReqs * sampleSizePerReq)
    {
        int reqIdx = wid % topkParam.numReqs;
        int docIdx = wid / topkParam.numReqs;
        dm_scoreSample[wid] = topkParam.dm_score[reqIdx * topkParam.numDocs + docIdx];
    }
}

void TopkSampling::sample(TopkParam &param)
{
    int blockSize = 256;
    int gridSize = (param.numReqs * kNumSamplesPerReq + blockSize - 1) / blockSize;
    sampleKernelNonRandom<<<gridSize, blockSize>>>(param, dm_scoreSample, kNumSamplesPerReq);
    CHECK_CUDA(cudaDeviceSynchronize());
}

__global__ void updateThreshold(float *dm_scoreSample, float *dm_scoreThreshold, int numReqs, int thIdx, size_t kNumSamplesPerReq)
{
    int reqIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (reqIdx < numReqs)
    {
        dm_scoreThreshold[reqIdx] = dm_scoreSample[reqIdx * kNumSamplesPerReq + thIdx];
    }
}

void TopkSampling::findThreshold(TopkParam &param)
{
    int thIdx = int((double)param.numToRetrieve / param.numDocs * kNumSamplesPerReq);
    for (size_t reqIdx = 0; reqIdx < param.numReqs; reqIdx++)
    {
        thrust::sort(thrust::device,
                     dm_scoreSample + reqIdx       * kNumSamplesPerReq,
                     dm_scoreSample + (reqIdx + 1) * kNumSamplesPerReq,
                     greater<float>());
    }

    int blockSize = 256;
    int gridSize = (param.numReqs + blockSize - 1) / blockSize;
    updateThreshold<<<gridSize, blockSize>>>(dm_scoreSample, dm_scoreThreshold, param.numReqs, thIdx, kNumSamplesPerReq);
}

__global__ void copyEligibleKernel(float *dm_score,
                                   float *dm_scoreThreshold,
                                   Pair *dm_eligiblePairs,
                                   int *dm_copyCount,
                                   int numReqs,
                                   int numDocs,
                                   size_t kMaxEligiblePairsPerDoc)
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
            if (count < kMaxEligiblePairsPerDoc)
            {
                Pair pair;
                pair.reqId = reqIdx;
                pair.docId = docIdx;
                pair.score = score;
                dm_eligiblePairs[reqIdx * kMaxEligiblePairsPerDoc + count] = pair;
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

    CHECK_CUDA(cudaMemset(dm_copyCount, 0, param.numReqs * sizeof(int)));
    int blockSize = 256;
    int gridSize = (param.numReqs * param.numDocs + blockSize - 1) / blockSize;
    copyEligibleKernel<<<gridSize, blockSize>>>(param.dm_score,
                                                dm_scoreThreshold,
                                                dm_eligiblePairs,
                                                dm_copyCount,
                                                param.numReqs,
                                                param.numDocs,
                                                kMaxEligiblePairsPerDoc);
}

void TopkSampling::retrieveExact(TopkParam &param)
{
    for (size_t reqIdx = 0; reqIdx < param.numReqs; reqIdx++)
    {
        Pair *dm_eligiblePairsStart = dm_eligiblePairs + reqIdx * kMaxEligiblePairsPerDoc;
        Pair *dm_eligiblePairsEnd = dm_eligiblePairsStart + dm_copyCount[reqIdx];
        thrust::sort(thrust::device,
                     dm_eligiblePairsStart,
                     dm_eligiblePairsEnd,
                     scoreComparator);
        thrust::copy(thrust::device,
                     dm_eligiblePairsStart,
                     dm_eligiblePairsStart + param.numToRetrieve,
                     param.dm_rst + reqIdx * param.numToRetrieve);
    }
}
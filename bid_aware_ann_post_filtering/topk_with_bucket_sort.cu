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

void Topk::init(float minScore, float maxScore)
{
    minScore_ = minScore;
    maxScore_ = maxScore;
    CHECK_CUDA(cudaMalloc(&d_counter_, kSize_byte_d_counter_));
    CHECK_CUDA(cudaMallocHost(&h_counter_, kSize_byte_d_counter_));
}

void Topk::reset()
{
    CHECK_CUDA(cudaFree(d_counter_));
    CHECK_CUDA(cudaFreeHost(h_counter_));
}

__global__ void updateCounterKernel(ReqDocPair *d_doc, int numReqDocPairs, Topk retriever)
{
    int docId = blockIdx.x * blockDim.x + threadIdx.x;

    if (docId < numReqDocPairs)
    {
        ReqDocPair doc = d_doc[docId];
        retriever.updateCounter(doc);
    }
}

void Topk::findLowestBucket(vector<int> &v_counter, int numToRetrieve, int &lowestBucket, int &numReqDocPairsGreaterThanLowestBucket)
{
    lowestBucket = 0;
    numReqDocPairsGreaterThanLowestBucket = 0;
    // Starting from the highest bucket, accumulate the count until it satisfies numToRetrieve
    for (int bucket = kGranularity_; bucket >= 0; bucket--)
    {
        // Accumulate the count of all slots into the first slot
        int slot0 = 0;
        int counterIdx0 = getCounterIdx(slot0, bucket);
        for (int slot = 1; slot < kNumSlots_; slot++)
        {
            int counterIdx = getCounterIdx(slot, bucket);
            v_counter[counterIdx0] += v_counter[counterIdx];
        }
        numReqDocPairsGreaterThanLowestBucket += v_counter[counterIdx0];
        if (numReqDocPairsGreaterThanLowestBucket >= numToRetrieve)
        {
            lowestBucket = bucket;
            break;
        }
    }
}

vector<ReqDocPair> Topk::retrieveTopk(TopkRetrievalParam &param)
{
    CudaTimer timer;
    CudaTimer timerLocal;
    timer.tic();

    int kBlockSize = 256;
    int gridSize = (int)ceil((double)(param.numReqDocPairs + 1) / kBlockSize);

    // Step1 - Run kernel to update the counter
    timerLocal.tic();
    CHECK_CUDA(cudaMemset(d_counter_, 0, kSize_byte_d_counter_))
    updateCounterKernel<<<gridSize, kBlockSize>>>(param.d_pair, param.numReqDocPairs, *this);
    cudaDeviceSynchronize();
    CHECK_CUDA(cudaGetLastError())
    param.timeMsUpdateCounter = timerLocal.tocMs();

    // Step2 - Copy counter from GPU to CPU
    timerLocal.tic();
    vector<int> v_counter(kSize_d_counter_, 0);
    CHECK_CUDA(cudaMemcpy(v_counter.data(), d_counter_, kSize_byte_d_counter_, cudaMemcpyDeviceToHost))
    param.timeMsCopyCounterToCpu = timerLocal.tocMs();

    // Step3 - Find the lowest bucket
    timerLocal.tic();
    int numReqDocPairsGreaterThanLowestBucket;
    findLowestBucket(v_counter, param.numToRetrieve, lowestBucket_, numReqDocPairsGreaterThanLowestBucket);
    param.timeMsFindLowestBucket = timerLocal.tocMs();

    // Step4 - Filter items that is larger than the lowest bucket
    timerLocal.tic();
    ReqDocPair *d_endPtr = thrust::copy_if(thrust::device, param.d_pair, param.d_pair + param.numReqDocPairs, param.d_buffer, *this); // copy_if will call Topk::operator()
    cudaDeviceSynchronize();
    CHECK_CUDA(cudaGetLastError())
    int numCopied = (d_endPtr - param.d_buffer);
    assert(numCopied == numReqDocPairsGreaterThanLowestBucket);
    param.timeMsPrefilter = timerLocal.tocMs();

    // Step5 - Only sort the docs that are larger than the lowest bucket
    timerLocal.tic();
    thrust::stable_sort(thrust::device, param.d_buffer, param.d_buffer + numCopied, ScorePredicator());
    cudaDeviceSynchronize();
    CHECK_CUDA(cudaGetLastError())
    param.timeMsSort = timerLocal.tocMs();

    // Step6 - copy back to CPU
    timerLocal.tic();
    int numToRetrieve = min(param.numToRetrieve, numCopied);
    vector<ReqDocPair> v_doc(numToRetrieve);
    CHECK_CUDA(cudaMemcpy(v_doc.data(), param.d_buffer, sizeof(ReqDocPair) * numToRetrieve, cudaMemcpyDeviceToHost))
    param.timeMsCopyBackToCpu = timerLocal.tocMs();

    param.timeMsTotal = timer.tocMs();

    return v_doc;
}

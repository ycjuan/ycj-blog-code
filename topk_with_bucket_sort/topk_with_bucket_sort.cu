#include <vector>
#include <cmath>
#include <cassert>
#include <thrust/copy.h>
#include <thrust/execution_policy.h>
#include <thrust/sort.h>

#include "topk.cuh"
#include "util.cuh"

void TopkBucketSort::init()
{
    CHECK_CUDA(cudaMalloc(&d_counter_, kNumBuckets_ * kNumSlots_ * sizeof(int)));
    CHECK_CUDA(cudaMallocHost(&h_counter_, kNumBuckets_ * kNumSlots_ * sizeof(int)));
}

void TopkBucketSort::reset()
{
    CHECK_CUDA(cudaFree(d_counter_));
    CHECK_CUDA(cudaFreeHost(h_counter_));
}

__global__ void updateCounterKernel(Doc *d_doc, int numDocs, TopkBucketSort retriever)
{
    int docId = blockIdx.x * blockDim.x + threadIdx.x;

    if (docId < numDocs)
    {
        Doc doc = d_doc[docId];
        retriever.updateCounter(doc);
    }
}

void TopkBucketSort::findLowestBucket(std::vector<int> &v_counter, int numToRetrieve, int &lowestBucket, int &numDocsGreaterThanLowestBucket)
{
    lowestBucket = 0;
    numDocsGreaterThanLowestBucket = 0;
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
        numDocsGreaterThanLowestBucket += v_counter[counterIdx0];
        if (numDocsGreaterThanLowestBucket >= numToRetrieve)
        {
            lowestBucket = bucket;
            break;
        }
    }
}

std::vector<Doc> TopkBucketSort::retrieveTopk(Doc *d_doc, Doc *d_buffer, int numDocs, int numToRetrieve, float &timeMs)
{
    Timer timer;
    timer.tic();

    int kBlockSize = 256;
    int gridSize = (int)ceil((double)(numDocs + 1) / kBlockSize);

    // Step1 - Run kernel to update the counter
    CHECK_CUDA(cudaMemset(d_counter_, 0, kSize_byte_d_counter_))
    updateCounterKernel<<<gridSize, kBlockSize>>>(d_doc, numDocs, *this);
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaGetLastError())

    // Step2 - Copy counter from GPU to CPU
    std::vector<int> v_counter(kSize_d_counter_, 0);
    CHECK_CUDA(cudaMemcpy(v_counter.data(), d_counter_, kSize_byte_d_counter_, cudaMemcpyDeviceToHost))

    // Step3 - Find the lowest bucket
    int numDocsGreaterThanLowestBucket;
    findLowestBucket(v_counter, numToRetrieve, lowestBucket_, numDocsGreaterThanLowestBucket);

    // Step4 - Filter items that is larger than the lowest bucket
    Doc *d_endPtr = thrust::copy_if(thrust::device, d_doc, d_doc + numDocs, d_buffer, *this); // copy_if will call TopkBucketSort::operator()
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaGetLastError())
    int numCopied = (d_endPtr - d_buffer);
    assert(numCopied == numDocsGreaterThanLowestBucket);

    // Step5 - Only sort the docs that are larger than the lowest bucket
    thrust::stable_sort(thrust::device, d_buffer, d_buffer + numCopied, ScorePredicator());
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaGetLastError())

    // Step6 - copy back to CPU
    std::vector<Doc> v_doc(numToRetrieve);
    CHECK_CUDA(cudaMemcpy(v_doc.data(), d_buffer, sizeof(Doc) * numToRetrieve, cudaMemcpyDeviceToHost))

    timeMs = timer.tocMs();

    return v_doc;
}

#pragma once

#include <vector>
#include <thrust/copy.h>
#include <thrust/execution_policy.h>
#include <thrust/sort.h>
#include <sstream>
#include <stdexcept>

#include "util.cuh"

struct Doc
{
    int docId;
    float score;
    bool operator==(const Doc& other) const { return docId == other.docId && score == other.score; }
};

inline __device__ __host__ bool scoreComparator(const Doc& a, const Doc& b) { return a.score > b.score; }

struct ScorePredicator
{
    inline __host__ __device__ bool operator()(const Doc& a, const Doc& b) { return scoreComparator(a, b); }
};

template <typename T> class TopkBucketSort; // forward declaration

template <typename T>
__global__ void updateCounterKernel(T* d_doc, int numDocs, TopkBucketSort<T> retriever)
{
    int docId = blockIdx.x * blockDim.x + threadIdx.x;

    if (docId < numDocs)
    {
        T doc = d_doc[docId];
        retriever.updateCounter(doc);
    }
}

template <typename T> class TopkBucketSort
{
public:
    void init()
    {

        CHECK_CUDA(cudaMalloc(&d_counter_, kNumBuckets_ * kNumSlots_ * sizeof(int)));
        CHECK_CUDA(cudaMallocHost(&h_counter_, kNumBuckets_ * kNumSlots_ * sizeof(int)));
    }

    void reset()
    {
        CHECK_CUDA(cudaFree(d_counter_));
        CHECK_CUDA(cudaFreeHost(h_counter_));
    }

    std::vector<T> retrieveTopk(T* d_doc, T* d_buffer, int numDocs, int numToRetrieve, float& timeMs)
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
        if (numCopied != numDocsGreaterThanLowestBucket)
        {
            std::ostringstream oss;
            oss << "numCopied != numDocsGreaterThanLowestBucket: " << numCopied << " != " << numDocsGreaterThanLowestBucket;
            throw std::runtime_error(oss.str());
        }
    
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

    __device__ __host__ int getCounterIdx(int slot, int bucket) { return slot * kNumBuckets_ + bucket; }

    __device__ int bucketize(float score)
    {
        score = min(kMaxScore_, max(kMinScore_, score));
        score = (score - kMinScore_) / (kMaxScore_ - kMinScore_);
        return (int)(score * kGranularity_);
    }

    __device__ void updateCounter(T doc)
    {
        int slot = doc.docId % kNumSlots_;
        int bucket = bucketize(doc.score);
        int counterIdx = getCounterIdx(slot, bucket);

        atomicAdd(&d_counter_[counterIdx], 1);
    }

    __device__ bool operator()(const T& doc)
    {
        int bucket = bucketize(doc.score);
        return bucket >= lowestBucket_;
    }

private:
    const float kMinScore_ = -1;
    const float kMaxScore_ = 1;
    const int kGranularity_ = 512;
    const int kNumBuckets_ = kGranularity_ + 1;
    // For example, if you have kMinScore = -1.0, kMaxScore = 1.0, kGranularity = 2,
    // then you will have 3 buckets: [-1, 0, 1]
    const int kNumSlots_ = 16;
    // Each bucket has 16 slots. A GPU thread will randomly write into one of the thread,
    // and then the count will finally be accumulated to the first slot.
    // The purpose of doing this is to minimize too much concurrent write to the same memory location with atomicAdd.
    const int kSize_d_counter_ = kNumBuckets_ * kNumSlots_;
    const int kSize_byte_d_counter_ = kSize_d_counter_ * sizeof(int);

    int* d_counter_ = nullptr;
    int* h_counter_ = nullptr;
    int lowestBucket_ = 0;

    void findLowestBucket(std::vector<int>& v_counter,
                          int numToRetrieve,
                          int& lowestBucket,
                          int& numDocsGreaterThanLowestBucket)
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
};

std::vector<Doc> retrieveTopkGpuFullSort(Doc* d_doc, int numDocs, int numToRetrieve, float& timeMs);
std::vector<Doc> retrieveTopkCpuFullSort(std::vector<Doc>& v_doc, int numToRetrieve, float& timeMs);
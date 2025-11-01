#pragma once

#include <random>
#include <vector>
#include <thrust/copy.h>
#include <thrust/execution_policy.h>
#include <thrust/sort.h>
#include <sstream>
#include <stdexcept>

#include "util.cuh"

template <typename T, class ScorePredicator, class DocIdExtractor, class ScoreExtractor> class TopkBucketSort; // forward declaration

template <typename T, class ScorePredicator, class DocIdExtractor, class ScoreExtractor>
__global__ void updateCounterKernel(T* d_doc, int numDocs, TopkBucketSort<T, ScorePredicator, DocIdExtractor, ScoreExtractor> retriever)
{
    int docId = blockIdx.x * blockDim.x + threadIdx.x;

    if (docId < numDocs)
    {
        T doc = d_doc[docId];
        retriever.updateCounter(doc);
    }
}

template<typename T, class ScoreExtractor>
__global__ void sampleRandomScoresKernel(T* d_doc, int numToSample, float* d_sampledScores, uint32_t* d_randomIndices)
{
    int docId = blockIdx.x * blockDim.x + threadIdx.x;
    if (docId < numToSample)
    {
        d_sampledScores[docId] = ScoreExtractor()(d_doc[d_randomIndices[docId]]);
    }
}

template <typename T, class ScorePredicator, class DocIdExtractor, class ScoreExtractor> class TopkBucketSort
{
public:
    void init()
    {
        shouldCheckMinMaxScore_ = true;
        CHECK_CUDA(cudaMalloc(&d_counter_, kNumBuckets_ * kNumSlots_ * sizeof(int)));
        CHECK_CUDA(cudaMallocHost(&hp_counter_, kNumBuckets_ * kNumSlots_ * sizeof(int)));
        CHECK_CUDA(cudaMalloc(&d_randomIndices_, kNumDocsToSample_ * sizeof(uint32_t)));
        CHECK_CUDA(cudaMallocHost(&hp_randomIndices_, kNumDocsToSample_ * sizeof(uint32_t)));
        CHECK_CUDA(cudaMalloc(&d_sampledScores_, kNumDocsToSample_ * sizeof(float)));
        CHECK_CUDA(cudaMallocHost(&hp_sampledScores_, kNumDocsToSample_ * sizeof(float)));
    }

    void init(float minScore, float maxScore)
    {
        minScore_ = minScore;
        maxScore_ = maxScore;
        init();
        shouldCheckMinMaxScore_ = false; // Since the caller has already provided the min and max score, we don't need to check again.
    }

    void reset()
    {
        CHECK_CUDA(cudaFree(d_counter_));
        CHECK_CUDA(cudaFreeHost(hp_counter_));
        CHECK_CUDA(cudaFree(d_randomIndices_));
        CHECK_CUDA(cudaFreeHost(hp_randomIndices_));
        CHECK_CUDA(cudaFree(d_sampledScores_));
        CHECK_CUDA(cudaFreeHost(hp_sampledScores_));
    }

    std::vector<T> retrieveTopk(T* d_doc, T* d_buffer, int numDocs, int numToRetrieve, float& timeMs)
    {
        Timer timer;
        timer.tic();
    
        int kBlockSize = 256;
        int gridSize = (int)ceil((double)(numDocs + 1) / kBlockSize);

        if (shouldCheckMinMaxScore_)
        {
            checkMinMaxScore(d_doc, numDocs);
            //std::cout << "Min score: " << minScore_ << ", Max score: " << maxScore_ << std::endl;
        }
    
        // Step1 - Run kernel to update the counter in each bucket
        CHECK_CUDA(cudaMemset(d_counter_, 0, kSize_byte_d_counter_))
        updateCounterKernel<<<gridSize, kBlockSize>>>(d_doc, numDocs, *this);
        CHECK_CUDA(cudaDeviceSynchronize());
        CHECK_CUDA(cudaGetLastError())
    
        // Step2 - Copy counter from GPU to CPU
        std::vector<int> v_counter(kSize_d_counter_, 0);
        CHECK_CUDA(cudaMemcpy(v_counter.data(), d_counter_, kSize_byte_d_counter_, cudaMemcpyDeviceToHost))
    
        // Step3 - Scan the bucket counter from high to low, and find the lowest bucket that has more docs than numToRetrieve
        int numDocsGreaterThanLowestBucket;
        findLowestBucket(v_counter, numToRetrieve, lowestBucket_, numDocsGreaterThanLowestBucket);
    
        // Step4 - Filter out items that is larger than the lowest bucket
        T *d_endPtr = thrust::copy_if(thrust::device, d_doc, d_doc + numDocs, d_buffer, *this); // copy_if will call TopkBucketSort::operator()
        CHECK_CUDA(cudaDeviceSynchronize());
        CHECK_CUDA(cudaGetLastError())
        int numCopied = (d_endPtr - d_buffer);
        if (numCopied != numDocsGreaterThanLowestBucket)
        {
            std::ostringstream oss;
            oss << "numCopied != numDocsGreaterThanLowestBucket: " << numCopied << " != " << numDocsGreaterThanLowestBucket;
            throw std::runtime_error(oss.str());
        }
    
        // Step5 - Sort the docs that are larger than the lowest bucket
        thrust::stable_sort(thrust::device, d_buffer, d_buffer + numCopied, ScorePredicator());
        CHECK_CUDA(cudaDeviceSynchronize());
        CHECK_CUDA(cudaGetLastError())
    
        // Step6 - copy back to CPU
        std::vector<T> v_doc(numToRetrieve);
        CHECK_CUDA(cudaMemcpy(v_doc.data(), d_buffer, sizeof(T) * numToRetrieve, cudaMemcpyDeviceToHost))
    
        timeMs = timer.tocMs();
    
        return v_doc;
    }

    // Note that each bucket has several slots, this is needed to avoid the contention of atomicAdd.
    // After the kernel is done, we will sum the count of all slots into the first slot, and then the first slot will be
    // the total count of the bucket.
    __device__ __host__ int getCounterIdx(int slot, int bucket) { return slot * kNumBuckets_ + bucket; }

    // Convert the score to the bucket index.
    __device__ int bucketize(float score)
    {
        score = min(maxScore_, max(minScore_, score));
        score = (score - minScore_) / (maxScore_ - minScore_);
        return (int)(score * kGranularity_);
    }

    __device__ void updateCounter(T doc)
    {
        int slot = DocIdExtractor()(doc) % kNumSlots_; // randomly write into one of the slots. This is to avoid the contention of atomicAdd.
        int bucket = bucketize(ScoreExtractor()(doc));
        int counterIdx = getCounterIdx(slot, bucket);

        atomicAdd(&d_counter_[counterIdx], 1);
    }

    // This function is used by thrust::copy_if in Step4.
    __device__ bool operator()(const T& doc)
    {
        int bucket = bucketize(ScoreExtractor()(doc));
        return bucket >= lowestBucket_;
    }

private:
    const int kGranularity_ = 512;
    const int kNumBuckets_ = kGranularity_ + 1;
    // For example, if we have kMinScore = -1.0, kMaxScore = 1.0, kGranularity = 2,
    // then you will have 3 buckets: [-1, 0, 1]
    const int kNumSlots_ = 16;
    // Each bucket has 16 slots. A GPU thread will randomly write into one of the thread,
    // and then the count will finally be accumulated to the first slot.
    // The purpose of doing this is to minimize too much concurrent write to the same memory location with atomicAdd.
    const int kSize_d_counter_ = kNumBuckets_ * kNumSlots_;
    const int kSize_byte_d_counter_ = kSize_d_counter_ * sizeof(int);

    // --------------
    // Sampling related Min and max score of the docs
    float minScore_ = -1;
    float maxScore_ = 1;
    bool shouldCheckMinMaxScore_ = true; // When this is set to true, the algorithm will run an additional step 
                                         // to sample some docs to get the min and max score.
    const uint32_t kNumDocsToSample_ = 10000;
    uint32_t* d_randomIndices_ = nullptr;
    uint32_t* hp_randomIndices_ = nullptr;
    float *d_sampledScores_ = nullptr;
    float *hp_sampledScores_ = nullptr;
    float maxPercentile_ = 0.99;
    float minPercentile_ = 0.01;

    int* d_counter_ = nullptr;
    int* hp_counter_ = nullptr;
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

    void checkMinMaxScore(T* d_doc, uint32_t numDocs)
    {
        // --------------
        // Step1 - Sample some random indices
        std::default_random_engine generator;
        std::uniform_int_distribution<uint32_t> distribution(0, numDocs - 1);
        for (int i = 0; i < kNumDocsToSample_; i++)
        {
            hp_randomIndices_[i] = distribution(generator);
        }

        // --------------
        // Step2 - Copy the random indices to GPU
        CHECK_CUDA(cudaMemcpy(d_randomIndices_, hp_randomIndices_, kNumDocsToSample_ * sizeof(uint32_t), cudaMemcpyHostToDevice));

        // --------------
        // Step3 - Sample the scores
        int kBlockSize = 256;
        int gridSize = (int)ceil((double)(numDocs + 1) / kBlockSize);
        int numToSample = std::min(numDocs, kNumDocsToSample_);
        sampleRandomScoresKernel<T, ScoreExtractor>
            <<<gridSize, kBlockSize>>>(d_doc, numToSample, d_sampledScores_, d_randomIndices_);
        CHECK_CUDA(cudaDeviceSynchronize());
        CHECK_CUDA(cudaGetLastError())

        // --------------
        // Step4 - Sort the scores in GPU
        thrust::sort(thrust::device, d_sampledScores_, d_sampledScores_ + kNumDocsToSample_, thrust::less<float>());

        // --------------
        // Step5 - Copy the scores back to CPU
        CHECK_CUDA(cudaMemcpy(hp_sampledScores_, d_sampledScores_, kNumDocsToSample_ * sizeof(float), cudaMemcpyDeviceToHost));

        minScore_ = hp_sampledScores_[(int)(kNumDocsToSample_ * minPercentile_)];
        maxScore_ = hp_sampledScores_[(int)(kNumDocsToSample_ * maxPercentile_)];
    }
};

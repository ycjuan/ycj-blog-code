#pragma once

#include <cstdint>
#include <random>
#include <vector>
#include <thrust/copy.h>
#include <thrust/execution_policy.h>
#include <thrust/sort.h>
#include <sstream>
#include <stdexcept>

#include "util.cuh"

template <typename T, class ScorePredicator, class DocIdExtractor, class ScoreExtractor> class TopkBucketSort; // forward declaration

// This kernel is used to update the counter in each bucket.
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

// This kernel is used to sample the scores from the docs.
template<typename T, class ScoreExtractor>
__global__ void sampleRandomScoresKernel(T* d_doc, uint32_t numToSample, float* d_sampledScores, uint32_t* d_randomIndices, uint32_t numDocs)
{
    int docIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (docIdx < numToSample)
    {
        d_sampledScores[docIdx] = ScoreExtractor()(d_doc[d_randomIndices[docIdx] % numDocs]);
    }
}

// This kernel is used to update the min and max score of the sampled scores.
__managed__ __device__ float g_minScore;
__managed__ __device__ float g_maxScore;
__global__ void updateMinMaxScoreKernel(float *d_sampledScores, int numSamples, float minPercentile, float maxPercentile)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid == 0)
    {
        g_minScore = d_sampledScores[(int)(numSamples * minPercentile)];
        g_maxScore = d_sampledScores[(int)(numSamples * maxPercentile)];
    }
}

template <typename T, class ScorePredicator, class DocIdExtractor, class ScoreExtractor> class TopkBucketSort
{
public:
    void init(uint64_t maxNumDocs)
    {
        maxNumDocs_ = maxNumDocs;
        size_byte_d_buffer_ = sizeof(T) * maxNumDocs;
        CHECK_CUDA(cudaMalloc(&d_buffer_, size_byte_d_buffer_));
        CHECK_CUDA(cudaMalloc(&d_counter_, kNumBuckets_ * kNumSlots_ * sizeof(int)));
        CHECK_CUDA(cudaMallocHost(&hp_counter_, kNumBuckets_ * kNumSlots_ * sizeof(int)));
        CHECK_CUDA(cudaMalloc(&d_randomIndices_, kNumDocsToSample_ * sizeof(uint32_t)));
        CHECK_CUDA(cudaMallocHost(&hp_randomIndices_, kNumDocsToSample_ * sizeof(uint32_t)));
        CHECK_CUDA(cudaMalloc(&d_sampledScores_, kNumDocsToSample_ * sizeof(float)));
        CHECK_CUDA(cudaMallocHost(&hp_sampledScores_, kNumDocsToSample_ * sizeof(float)));

        // Initialize the random indices.
        // We do this in init time as it is expensive to generate random indices in runtime.
        // In runtime, we will do `% numDocs` to get the final random index.
        std::default_random_engine generator;
        std::uniform_int_distribution<uint32_t> distribution;
        for (int i = 0; i < kNumDocsToSample_; i++)
        {
            hp_randomIndices_[i] = distribution(generator);
        }
        CHECK_CUDA(cudaMemcpy(d_randomIndices_, hp_randomIndices_, kNumDocsToSample_ * sizeof(uint32_t), cudaMemcpyHostToDevice));
    }

    void reset()
    {
        CHECK_CUDA(cudaFree(d_buffer_));
        CHECK_CUDA(cudaFree(d_counter_));
        CHECK_CUDA(cudaFreeHost(hp_counter_));
        CHECK_CUDA(cudaFree(d_randomIndices_));
        CHECK_CUDA(cudaFreeHost(hp_randomIndices_));
        CHECK_CUDA(cudaFree(d_sampledScores_));
        CHECK_CUDA(cudaFreeHost(hp_sampledScores_));
    }

    std::vector<T> retrieveTopk(T* d_doc, int numDocs, int numToRetrieve, float& timeMs)
    {
        Timer timer;
        timer.tic();
    
        int kBlockSize = 256;
        int gridSize = (int)ceil((double)(numDocs + 1) / kBlockSize);

        // --------------
        // Step1 (Optional) - Check the min and max score of the docs if those values are not set by the user.
        if (shouldCheckMinMaxScore_)
        {
            Timer timerStep1;
            timerStep1.tic();
            checkMinMaxScore(d_doc, numDocs);
            float timeMsStep1 = timerStep1.tocMs();
            std::cout << "[TopkBucketSort::retrieveTopk] timeMsStep1: " << timeMsStep1 << " ms" << std::endl;
        }
    
        // --------------
        // Step2 - Run kernel to update the counter in each bucket
        {
            Timer timerStep2;
            timerStep2.tic();
            CHECK_CUDA(cudaMemset(d_counter_, 0, kSize_byte_d_counter_))
            updateCounterKernel<<<gridSize, kBlockSize>>>(d_doc, numDocs, *this);
            CHECK_CUDA(cudaDeviceSynchronize());
            CHECK_CUDA(cudaGetLastError());
            float timeMsStep2 = timerStep2.tocMs();
            std::cout << "[TopkBucketSort::retrieveTopk] timeMsStep2: " << timeMsStep2 << " ms" << std::endl;
        }
    
        // --------------
        // Step3 - Copy counter from GPU to CPU
        std::vector<int> v_counter(kSize_d_counter_, 0);
        {
            Timer timerStep3;
            timerStep3.tic();
            CHECK_CUDA(cudaMemcpy(v_counter.data(), d_counter_, kSize_byte_d_counter_, cudaMemcpyDeviceToHost))
            float timeMsStep3 = timerStep3.tocMs();
            std::cout << "[TopkBucketSort::retrieveTopk] timeMsStep3: " << timeMsStep3 << " ms" << std::endl;
        }
        
        // --------------
        // Step4 - Scan the bucket counter from high to low, and find the lowest bucket that has more docs than numToRetrieve
        int numDocsGreaterThanLowestBucket;
        {
            Timer timerStep4;
            timerStep4.tic();
            findLowestBucket(v_counter, numToRetrieve, lowestBucket_, numDocsGreaterThanLowestBucket);
            float timeMsStep4 = timerStep4.tocMs();
            std::cout << "[TopkBucketSort::retrieveTopk] timeMsStep4: " << timeMsStep4 << " ms" << std::endl;
        }
    
        // --------------
        // Step5 - Filter out items that is larger than the lowest bucket
        int numCopied;
        {
            Timer timerStep5;
            timerStep5.tic();
            T *d_endPtr = thrust::copy_if(thrust::device, d_doc, d_doc + numDocs, d_buffer_, *this); // copy_if will call TopkBucketSort::operator()
            CHECK_CUDA(cudaDeviceSynchronize());
            CHECK_CUDA(cudaGetLastError())
            numCopied = (d_endPtr - d_buffer_);
            if (numCopied != numDocsGreaterThanLowestBucket)
            {
                std::ostringstream oss;
                oss << "numCopied != numDocsGreaterThanLowestBucket: " << numCopied << " != " << numDocsGreaterThanLowestBucket;
                throw std::runtime_error(oss.str());
            }
            float timeMsStep5 = timerStep5.tocMs();
            std::cout << "[TopkBucketSort::retrieveTopk] timeMsStep5: " << timeMsStep5 << " ms" << std::endl;
        }
    
        // --------------
        // Step6 - Sort the docs that are larger than the lowest bucket
        {
            Timer timerStep6;
            timerStep6.tic();
            thrust::stable_sort(thrust::device, d_buffer_, d_buffer_ + numCopied, ScorePredicator());
            CHECK_CUDA(cudaDeviceSynchronize());
            CHECK_CUDA(cudaGetLastError())
            float timeMsStep6 = timerStep6.tocMs();
            std::cout << "[TopkBucketSort::retrieveTopk] timeMsStep6: " << timeMsStep6 << " ms" << std::endl;
        }

        // --------------
        // Step7 - copy back to CPU
        std::vector<T> v_doc(numToRetrieve);
        {
            Timer timerStep7;
            timerStep7.tic();
            CHECK_CUDA(cudaMemcpy(v_doc.data(), d_buffer_, sizeof(T) * numToRetrieve, cudaMemcpyDeviceToHost))
            float timeMsStep7 = timerStep7.tocMs();
            std::cout << "[TopkBucketSort::retrieveTopk] timeMsStep7: " << timeMsStep7 << " ms" << std::endl;
        }
    
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

    void setMinMaxScore(float minScore, float maxScore)
    {
        minScore_ = minScore;
        maxScore_ = maxScore;
        shouldCheckMinMaxScore_ = false;
    }

    void unsetMinMaxScore()
    {
        shouldCheckMinMaxScore_ = true;
    }

private:
    const int kGranularity_ = 512;
    const int kNumBuckets_ = kGranularity_ + 1;
    // For example, if we have kMinScore = -1.0, kMaxScore = 1.0, kGranularity = 2,
    // then you will have 3 buckets: [-1, 0, 1]
    const int kNumSlots_ = 16;
    // Each bucket has 16 slots. A GPU thread will randomly write into one of the thread,
    // and then the count will finally be accumulated to the first slot.
    // The purpose of doing this is to minimize contention of atomicAdd.
    const int kSize_d_counter_ = kNumBuckets_ * kNumSlots_;
    const int kSize_byte_d_counter_ = kSize_d_counter_ * sizeof(int);

    // --------------
    // Sampling related min and max score of the docs
    float minScore_;
    float maxScore_;
    bool shouldCheckMinMaxScore_ = true; // When this is set to true, the algorithm will run an additional step 
                                         // to sample some docs to get the min and max score.
    const uint32_t kNumDocsToSample_ = 10000;
    uint32_t* d_randomIndices_ = nullptr;
    uint32_t* hp_randomIndices_ = nullptr;
    float* d_sampledScores_ = nullptr;
    float* hp_sampledScores_ = nullptr;
    float maxPercentile_ = 0.999;
    float minPercentile_ = 0.001;

    // ------------
    // Max num docs and buffer size
    T* d_buffer_ = nullptr;
    int maxNumDocs_ = 0;
    uint64_t size_byte_d_buffer_ = 0;

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
        int numToSample = std::min(numDocs, kNumDocsToSample_);

        // --------------
        // Step1 - Sample the scores
        {
            Timer timerStep1;
            timerStep1.tic();
            int kBlockSize = 256;
            int gridSize = (int)ceil((double)(numDocs + 1) / kBlockSize);
            sampleRandomScoresKernel<T, ScoreExtractor>
                <<<gridSize, kBlockSize>>>(d_doc, numToSample, d_sampledScores_, d_randomIndices_, numDocs);
            CHECK_CUDA(cudaDeviceSynchronize());
            CHECK_CUDA(cudaGetLastError())
            float timeMsStep1 = timerStep1.tocMs();
            std::cout << "timeMsStep1: " << timeMsStep1 << " ms" << std::endl;    
        }

        // --------------
        // Step2 - Sort the scores in GPU
        {
            Timer timerStep2;
            timerStep2.tic();
            thrust::sort(thrust::device, d_sampledScores_, d_sampledScores_ + kNumDocsToSample_, thrust::less<float>());
            float timeMsStep2 = timerStep2.tocMs();
            std::cout << "timeMsStep2: " << timeMsStep2 << " ms" << std::endl;
        }


        // --------------
        // Step3 - Update the min and max score
        {
            Timer timerStep3;
            timerStep3.tic();
            updateMinMaxScoreKernel<<<1, 1>>>(d_sampledScores_, numToSample, minPercentile_, maxPercentile_);
            CHECK_CUDA(cudaDeviceSynchronize());
            //CHECK_CUDA(cudaGetLastError())
            float timeMsStep3 = timerStep3.tocMs();
            std::cout << "timeMsStep3: " << timeMsStep3 << " ms" << std::endl;
        }

        // --------------
        // Step4 - Update the min and max score in CPU
        {
            minScore_ = g_minScore;
            maxScore_ = g_maxScore;
        }
    }
};

#pragma once

#include <random>
#include <sstream>
#include <stdexcept>
#include <thrust/copy.h>
#include <thrust/execution_policy.h>
#include <thrust/sort.h>
#include <vector>

#include "util.cuh"

namespace topk_gpu_sampling
{

__device__ float d_scoreThreshold_ = 0.0f; // Threshold of the scores that will be enough to satisfy numToRetrieve

// This kernel is used to sample the scores from the docs.
template <typename T, class ScoreExtractor>
__global__ void
sampleRandomScoresKernel(T* d_doc, int numToSample, float* d_sampledScores, int* d_randomIndices, int numDocs)
{
    int docIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (docIdx < numToSample)
    {
        d_sampledScores[docIdx] = ScoreExtractor()(d_doc[d_randomIndices[docIdx] % numDocs]);
    }
}
// This kernel is used to update the threshold of the sampled scores.
__global__ void
updateThresholdKernel(float* d_sampledScores, int targetSampleIdx)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid == 0)
    {
        d_scoreThreshold_ = d_sampledScores[targetSampleIdx];
    }
}

template <typename T, class ScoreExtractor>
class ThresholdPredicate
{
public:
    ThresholdPredicate(float scoreThreshold) : scoreThreshold_(scoreThreshold) {}

    __device__ bool operator()(const T& doc)
    {
        return ScoreExtractor()(doc) > scoreThreshold_;
    }

private:
    float scoreThreshold_;
};

}

template <typename T, class ScorePredicator, class DocIdExtractor, class ScoreExtractor> class TopkSampling
{
public:
    void init(size_t maxNumDocs)
    {
        maxNumDocs_ = maxNumDocs;
        size_byte_d_buffer_ = sizeof(T) * maxNumDocs;
        CHECK_CUDA(cudaMalloc(&d_buffer_, size_byte_d_buffer_));
        CHECK_CUDA(cudaMalloc(&d_randomIndices_, kNumDocsToSample_ * sizeof(int)));
        CHECK_CUDA(cudaMalloc(&d_sampledScores_, kNumDocsToSample_ * sizeof(float)));

        // Initialize the random indices.
        // We do this in init time as it is expensive to generate random indices in runtime.
        // In runtime, we will do `% numDocs` to get the final random index.
        std::default_random_engine generator;
        std::uniform_int_distribution<int> distribution(0, maxNumDocs_ - 1);
        std::vector<int> v_randomIndices(kNumDocsToSample_);
        for (int i = 0; i < kNumDocsToSample_; i++)
        {
            v_randomIndices[i] = distribution(generator);
        }
        CHECK_CUDA(
            cudaMemcpy(d_randomIndices_, v_randomIndices.data(), kNumDocsToSample_ * sizeof(int), cudaMemcpyHostToDevice));
    }

    void reset()
    {
        CHECK_CUDA(cudaFree(d_buffer_));
        CHECK_CUDA(cudaFree(d_randomIndices_));
        CHECK_CUDA(cudaFree(d_sampledScores_));
    }

    std::vector<T> retrieveTopk(T* d_doc, int numDocs, int numToRetrieve, cudaStream_t stream = nullptr)
    {
        // --------------
        // Step1 - Find the threshold of the scores
        {
            Timer timerStep1;
            timerStep1.tic();
            findThreshold(d_doc, numDocs, numToRetrieve, stream);
            float timeMsStep1 = timerStep1.tocMs();
            if (kPrintSegmentTime_)
            {
                std::cout << "[TopkBucketSort::retrieveTopk] timeMsStep1: " << timeMsStep1 << " ms" << std::endl;
            }
        }

        // --------------
        // Step2 - Filter out items that are larger than the threshold
        int numCopied;
        {
            Timer timerStep2;
            timerStep2.tic();
            T* d_endPtr = thrust::copy_if(thrust::device.on(stream), d_doc, d_doc + numDocs, d_buffer_,
                                          topk_gpu_sampling::ThresholdPredicate<T, ScoreExtractor>(h_scoreThreshold_));
            CHECK_CUDA(cudaStreamSynchronize(stream));
            CHECK_CUDA(cudaGetLastError());
            numCopied = (d_endPtr - d_buffer_);
            float timeMsStep2 = timerStep2.tocMs();
            if (kPrintSegmentTime_)
            {
                std::cout << "[TopkBucketSort::retrieveTopk] timeMsStep2: " << timeMsStep2 << " ms" << std::endl;
            }
        }

        // --------------
        // Step3 - Sort the docs that are larger than the threshold
        {
            Timer timerStep3;
            timerStep3.tic();
            thrust::stable_sort(thrust::device.on(stream), d_buffer_, d_buffer_ + numCopied, ScorePredicator());
            CHECK_CUDA(cudaStreamSynchronize(stream));
            CHECK_CUDA(cudaGetLastError());
            float timeMsStep3 = timerStep3.tocMs();
            if (kPrintSegmentTime_)
            {
                std::cout << "[TopkBucketSort::retrieveTopk] timeMsStep3: " << timeMsStep3 << " ms" << std::endl;
            }
        }

        // --------------
        // Step4 - copy back to CPU
        std::vector<T> v_doc(numToRetrieve);
        {
            Timer timerStep4;
            timerStep4.tic();
            CHECK_CUDA(cudaMemcpy(v_doc.data(), d_buffer_, sizeof(T) * numToRetrieve, cudaMemcpyDeviceToHost));
            CHECK_CUDA(cudaStreamSynchronize(stream));
            CHECK_CUDA(cudaGetLastError());
            float timeMsStep4 = timerStep4.tocMs();
            if (kPrintSegmentTime_)
            {
                std::cout << "[TopkBucketSort::retrieveTopk] timeMsStep4: " << timeMsStep4 << " ms" << std::endl;
            }
        }

        return v_doc;
    }

private:

    const int kNumDocsToSample_ = 10000; // Number of docs to sample to get the min and max score.
    const float kOverSamplingRatio_ = 1.0f;
    const bool kPrintSegmentTime_ = true;

    // ------------
    // Max num docs and buffer array
    T* d_buffer_ = nullptr; // This is needed for copy_if
    int maxNumDocs_ = 0;
    size_t size_byte_d_buffer_ = 0;

    // --------------
    // For sampling
    int* d_randomIndices_ = nullptr; // Random indices to sample the docs.
    float* d_sampledScores_ = nullptr; // Sampled scores from the docs.
    float h_scoreThreshold_ = 0.0f;

    void findThreshold(T* d_doc, int numDocs, int numToRetrieve, cudaStream_t stream)
    {
        int numToSample = std::min(numDocs, kNumDocsToSample_);

        // --------------
        // Step1 - Sample the scores
        {
            Timer timerStep1;
            timerStep1.tic();
            int kBlockSize = 256;
            int gridSize = (int)ceil((double)(numDocs + 1) / kBlockSize);
            topk_gpu_sampling::sampleRandomScoresKernel<T, ScoreExtractor>
                <<<gridSize, kBlockSize, 0, stream>>>(d_doc, numToSample, d_sampledScores_, d_randomIndices_, numDocs);
            CHECK_CUDA(cudaStreamSynchronize(stream));
            CHECK_CUDA(cudaGetLastError())
            float timeMsStep1 = timerStep1.tocMs();
            if (kPrintSegmentTime_)
            {
                std::cout << "[TopkSampling::findThreshold] timeMsStep1: " << timeMsStep1 << " ms" << std::endl;
            }
        }

        // --------------
        // Step2 - Sort the scores in GPU
        {
            Timer timerStep2;
            timerStep2.tic();
            thrust::sort(thrust::device.on(stream), d_sampledScores_, d_sampledScores_ + kNumDocsToSample_,
                         thrust::less<float>());
            float timeMsStep2 = timerStep2.tocMs();
            if (kPrintSegmentTime_)
            {
                std::cout << "[TopkSampling::findThreshold] timeMsStep2: " << timeMsStep2 << " ms" << std::endl;
            }
        }

        // --------------
        // Step3 - Update the threshold
        {
            Timer timerStep3;
            timerStep3.tic();
            int targetSampleIdx = (int)(numToSample * (numToRetrieve / (float)numDocs) * (1.0 + kOverSamplingRatio_));
            topk_gpu_sampling::updateThresholdKernel<<<1, 1, 0, stream>>>(d_sampledScores_, targetSampleIdx);
            CHECK_CUDA(cudaStreamSynchronize(stream));
            CHECK_CUDA(cudaGetLastError());
            float timeMsStep3 = timerStep3.tocMs();
            if (kPrintSegmentTime_)
            {
                std::cout << "[TopkSampling::findThreshold] timeMsStep3: " << timeMsStep3 << " ms" << std::endl;
            }
        }

        // --------------
        // Step4 - Copy the threshold from GPU to CPU
        {
            Timer timerStep4;
            timerStep4.tic();
            CHECK_CUDA(cudaMemcpyAsync(&h_scoreThreshold_, &topk_gpu_sampling::d_scoreThreshold_, sizeof(float), cudaMemcpyDeviceToHost, stream));
            CHECK_CUDA(cudaStreamSynchronize(stream));
            CHECK_CUDA(cudaGetLastError());
            float timeMsStep4 = timerStep4.tocMs();
            if (kPrintSegmentTime_)
            {
                std::cout << "[TopkSampling::findThreshold] timeMsStep4: " << timeMsStep4 << " ms" << std::endl;
            }
        }
    }
};

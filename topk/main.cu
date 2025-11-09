#include <iostream>
#include <random>
#include <stdexcept>
#include <vector>

#include "topk_cpu.cuh"
#include "topk_gpu_bucket.cuh"
#include "topk_gpu_sampling.cuh"
#include "util.cuh"

struct Doc
{
    int docId;
    float score;
    bool operator==(const Doc& other) const { return docId == other.docId && score == other.score; }
};

struct ScorePredicator
{
    inline __host__ __device__ bool operator()(const Doc& a, const Doc& b) { return a.score > b.score; }
};

struct DocIdExtractor
{
    inline __host__ __device__ int operator()(const Doc& doc) { return doc.docId; }
};

struct ScoreExtractor
{
    inline __host__ __device__ float operator()(const Doc& doc) { return doc.score; }
};

int kNumToRetrieve = 1000;
int kNumTrials = 10;

void runExp(int numDocs)
{
    std::cout << "\n\nrunning exps with numDocs: " << numDocs << std::endl;

    // --------------
    // Generate random data
    std::default_random_engine generator;
    std::uniform_real_distribution<float> distribution(-1.0, 1.0);
    std::vector<Doc> v_doc(numDocs);
    for (int i = 0; i < numDocs; i++)
    {
        v_doc[i].docId = i;
        v_doc[i].score = distribution(generator);
    }

    // --------------
    // Run CPU baseline
    std::vector<Doc> v_doc_copy = v_doc;
    Timer timerCpuFullSort;
    timerCpuFullSort.tic();
    std::vector<Doc> v_topk_cpuFullSort = retrieveTopkCpuFullSort<Doc, ScorePredicator>(v_doc_copy, kNumToRetrieve);
    float timeMsCpuFullSort = timerCpuFullSort.tocMs();

    // --------------
    // Init bucket topk
    TopkBucketSort<Doc, ScorePredicator, DocIdExtractor, ScoreExtractor> topkRetriever;
    topkRetriever.init(numDocs);
    Doc* d_doc = nullptr;
    CHECK_CUDA(cudaMalloc(&d_doc, numDocs * sizeof(Doc)));

    // --------------
    // Init sampling topk
    TopkSampling<Doc, ScorePredicator, DocIdExtractor, ScoreExtractor> topkSampler;
    topkSampler.init(numDocs);

    // --------------
    double timeMsGpuFullSort = 0;
    double timeMsGpuBucketSortWithSample = 0;
    double timeMsGpuBucketSortWithoutSample = 0;
    double timeMsGpuSampling = 0;
    for (int t = -3; t < kNumTrials; t++)
    {
        // --------------
        // Run GPU full sort
        {
            // ---------
            // Copy data to GPU
            CHECK_CUDA(cudaMemcpy(d_doc, v_doc.data(), numDocs * sizeof(Doc), cudaMemcpyHostToDevice));

            // ---------
            // Run GPU full sort
            Timer timerGpuFullSort;
            timerGpuFullSort.tic();
            std::vector<Doc> v_topk_gpuFullSort
                = retrieveTopkGpuFullSort<Doc, ScorePredicator>(d_doc, numDocs, kNumToRetrieve);
            float timeMsGpuFullSort1 = timerGpuFullSort.tocMs();
            if (t >= 0)
            {
                timeMsGpuFullSort += timeMsGpuFullSort1;
            }

            // ---------
            // Compare results
            if (v_topk_cpuFullSort != v_topk_gpuFullSort)
            {
                throw std::runtime_error("Topk results from CPU full sort and GPU full sort do not match");
            }
        }

        // --------------
        // Run GPU bucket sort with sample
        {
            // ---------
            // Copy data to GPU
            CHECK_CUDA(cudaMemcpy(d_doc, v_doc.data(), numDocs * sizeof(Doc), cudaMemcpyHostToDevice));

            // ---------
            // Run GPU bucket sort with sample
            Timer timerGpuBucketSortWithSample;
            timerGpuBucketSortWithSample.tic();
            topkRetriever.unsetMinMaxScore(); // unset min and max score so that the algorithm will perform sampling to
                                              // get the min and max score.
            std::vector<Doc> v_topk_gpuBucketSortWithSample
                = topkRetriever.retrieveTopk(d_doc, numDocs, kNumToRetrieve);
            float timeMsGpuBucketSortWithSample1 = timerGpuBucketSortWithSample.tocMs();
            if (t >= 0)
            {
                timeMsGpuBucketSortWithSample += timeMsGpuBucketSortWithSample1;
            }

            // ---------
            // Compare results
            if (v_topk_gpuBucketSortWithSample != v_topk_cpuFullSort)
            {
                throw std::runtime_error(
                    "Topk results from GPU bucket sort with sample and CPU full sort do not match");
            }
        }

        // --------------
        // Run GPU bucket sort without sample
        {
            // ---------
            // Copy data to GPU
            CHECK_CUDA(cudaMemcpy(d_doc, v_doc.data(), numDocs * sizeof(Doc), cudaMemcpyHostToDevice));

            // ---------
            // Run GPU bucket sort without sample
            Timer timerGpuBucketSortWithoutSample;
            timerGpuBucketSortWithoutSample.tic();
            topkRetriever.setMinMaxScore(-1.0, 1.0); // set min and max score so that the algorithm will directly use
                                                     // the min and max score. (and the sampling step will be skipped)
            std::vector<Doc> v_topk_gpuBucketSortWithoutSample
                = topkRetriever.retrieveTopk(d_doc, numDocs, kNumToRetrieve);
            float timeMsGpuBucketSortWithoutSample1 = timerGpuBucketSortWithoutSample.tocMs();
            if (t >= 0)
            {
                timeMsGpuBucketSortWithoutSample += timeMsGpuBucketSortWithoutSample1;
            }

            // ---------
            // Compare results
            if (v_topk_gpuBucketSortWithoutSample != v_topk_cpuFullSort)
            {
                throw std::runtime_error(
                    "Topk results from GPU bucket sort without sample and CPU full sort do not match");
            }
        }

        // --------------
        // Run GPU sampling topk
        {
            // ---------
            // Copy data to GPU
            CHECK_CUDA(cudaMemcpy(d_doc, v_doc.data(), numDocs * sizeof(Doc), cudaMemcpyHostToDevice));

            // ---------
            // Run GPU sampling topk
            Timer timerGpuSampling;
            timerGpuSampling.tic();
            std::vector<Doc> v_topk_gpuSampling = topkSampler.retrieveTopk(d_doc, numDocs, kNumToRetrieve);
            float timeMsGpuSampling1 = timerGpuSampling.tocMs();
            if (t >= 0)
            {
                timeMsGpuSampling += timeMsGpuSampling1;
            }

            // ---------
            // Compare results
            if (v_topk_gpuSampling != v_topk_cpuFullSort)
            {
                throw std::runtime_error("Topk results from GPU sampling topk and CPU full sort do not match");
            }
        }
    }

    // --------------
    // Print results
    timeMsGpuFullSort /= kNumTrials;
    timeMsGpuBucketSortWithSample /= kNumTrials;
    timeMsGpuBucketSortWithoutSample /= kNumTrials;
    timeMsGpuSampling /= kNumTrials;
    std::cout << "timeMsCpuFullSort: " << timeMsCpuFullSort << " ms" << std::endl;
    std::cout << "timeMsGpuFullSort: " << timeMsGpuFullSort << " ms" << std::endl;
    std::cout << "timeMsGpuBucketSortWithSample: " << timeMsGpuBucketSortWithSample << " ms" << std::endl;
    std::cout << "timeMsGpuBucketSortWithoutSample: " << timeMsGpuBucketSortWithoutSample << " ms" << std::endl;
    std::cout << "timeMsGpuSampling: " << timeMsGpuSampling << " ms" << std::endl;

    // --------------
    // Cleanup
    CHECK_CUDA(cudaFree(d_doc));
    topkRetriever.reset();
}

int main()
{
    runExp(1000000);
    runExp(32000000);

    return 0;
}
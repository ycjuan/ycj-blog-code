#include <random>
#include <vector>
#include <stdexcept>
#include <iostream>

#include "topk_cpu.cuh"
#include "topk_gpu.cuh"
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
    float timeMsCpuFullSort = 0;
    std::vector<Doc> v_doc_copy = v_doc;
    std::vector<Doc> v_topk_cpuFullSort = retrieveTopkCpuFullSort<Doc, ScorePredicator>(v_doc_copy, kNumToRetrieve, timeMsCpuFullSort);

    // --------------
    TopkBucketSort<Doc, ScorePredicator, DocIdExtractor, ScoreExtractor> topkRetriever;
    topkRetriever.init(numDocs);
    Doc *d_doc = nullptr;
    CHECK_CUDA(cudaMalloc(&d_doc, numDocs * sizeof(Doc)));

    double timeMsGpuFullSort = 0;
    double timeMsGpuBucketSortWithSample = 0;
    double timeMsGpuBucketSortWithoutSample = 0;
    for (int t = -3; t < kNumTrials; t++)
    {
        // --------------
        // Run GPU full sort
        {
            float timeMsGpuFullSort1 = 0;
            CHECK_CUDA(cudaMemcpy(d_doc, v_doc.data(), numDocs * sizeof(Doc), cudaMemcpyHostToDevice));
            std::vector<Doc> v_topk_gpuFullSort = retrieveTopkGpuFullSort<Doc, ScorePredicator>(d_doc, numDocs, kNumToRetrieve, timeMsGpuFullSort1);
            if (t >= 0)
            {
                timeMsGpuFullSort += timeMsGpuFullSort1;
            }

            if (v_topk_cpuFullSort != v_topk_gpuFullSort)
            {
                throw std::runtime_error("Topk results from CPU full sort and GPU full sort do not match");
            }    
        }

        // --------------
        // Run GPU bucket sort with sample
        {
            topkRetriever.unsetMinMaxScore();
            float timeMsGpuBucketSortWithSample1 = 0;
            CHECK_CUDA(cudaMemcpy(d_doc, v_doc.data(), numDocs * sizeof(Doc), cudaMemcpyHostToDevice));
            std::vector<Doc> v_topk_gpuBucketSortWithSample = topkRetriever.retrieveTopk(d_doc, numDocs, kNumToRetrieve, timeMsGpuBucketSortWithSample1);
            if (t >= 0)
            {
                timeMsGpuBucketSortWithSample += timeMsGpuBucketSortWithSample1;
            }
            
            if (v_topk_gpuBucketSortWithSample != v_topk_cpuFullSort)
            {
                throw std::runtime_error("Topk results from GPU bucket sort with sample and CPU full sort do not match");
            }
        }

        // --------------
        // Run GPU bucket sort without sample
        {
            topkRetriever.setMinMaxScore(-1.0, 1.0);
            float timeMsGpuBucketSortWithoutSample1 = 0;
            CHECK_CUDA(cudaMemcpy(d_doc, v_doc.data(), numDocs * sizeof(Doc), cudaMemcpyHostToDevice));
            std::vector<Doc> v_topk_gpuBucketSortWithoutSample = topkRetriever.retrieveTopk(d_doc, numDocs, kNumToRetrieve, timeMsGpuBucketSortWithoutSample1);
            if (t >= 0)
            {
                timeMsGpuBucketSortWithoutSample += timeMsGpuBucketSortWithoutSample1;
            }
            
            if (v_topk_gpuBucketSortWithoutSample != v_topk_cpuFullSort)
            {
                throw std::runtime_error("Topk results from GPU bucket sort without sample and CPU full sort do not match");
            }
        }
    }

    timeMsGpuFullSort /= kNumTrials;
    timeMsGpuBucketSortWithSample /= kNumTrials;
    timeMsGpuBucketSortWithoutSample /= kNumTrials;
    std::cout << "timeMsCpuFullSort: " << timeMsCpuFullSort << " ms" << std::endl;
    std::cout << "timeMsGpuFullSort: " << timeMsGpuFullSort << " ms" << std::endl;
    std::cout << "timeMsGpuBucketSortWithSample: " << timeMsGpuBucketSortWithSample << " ms" << std::endl;
    std::cout << "timeMsGpuBucketSortWithoutSample: " << timeMsGpuBucketSortWithoutSample << " ms" << std::endl;

    CHECK_CUDA(cudaFree(d_doc));
    topkRetriever.reset();
}

int main()
{
    runExp(1000000);
    runExp(32000000);

    return 0;
}
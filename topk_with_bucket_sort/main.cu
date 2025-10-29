#include <random>
#include <vector>
#include <stdexcept>
#include <iostream>

#include "topk.cuh"
#include "util.cuh"

int kNumToRetrieve = 1000;
int kNumTrials = 10;

void runExp(int numDocs)
{
    std::cout << "\n\nrunning exps with numDocs: " << numDocs << std::endl;
    std::default_random_engine generator;
    std::uniform_real_distribution<float> distribution(-1.0, 1.0);

    std::vector<Doc> v_doc(numDocs);
    for (int i = 0; i < numDocs; i++)
    {
        v_doc[i].docId = i;
        v_doc[i].score = distribution(generator);
    }

    TopkBucketSort retriever;
    retriever.init();
    Doc *d_doc = nullptr;
    Doc *d_buffer = nullptr;
    CHECK_CUDA(cudaMalloc(&d_doc, numDocs * sizeof(Doc)));
    CHECK_CUDA(cudaMalloc(&d_buffer, numDocs * sizeof(Doc)));

    double timeMsGpuBucketSort = 0;
    double timeMsGpuFullSort = 0;
    double timeMsCpuFullSort = 0;
    for (int t = -3; t < kNumTrials; t++)
    {
        float timeMsGpuBucketSort1 = 0;
        float timeMsGpuFullSort1 = 0;
        float timeMsCpuFullSort1 = 0;
        CHECK_CUDA(cudaMemcpy(d_doc, v_doc.data(), numDocs * sizeof(Doc), cudaMemcpyHostToDevice));
        std::vector<Doc> v_topk_gpuBucketSort = retriever.retrieveTopk(d_doc, d_buffer, numDocs, kNumToRetrieve, timeMsGpuBucketSort1);
        CHECK_CUDA(cudaMemcpy(d_doc, v_doc.data(), numDocs * sizeof(Doc), cudaMemcpyHostToDevice));
        std::vector<Doc> v_topk_gpuFullSort = retrieveTopkGpuFullSort(d_doc, numDocs, kNumToRetrieve, timeMsGpuFullSort1);
        std::vector<Doc> v_doc_copy = v_doc;
        std::vector<Doc> v_topk_cpuFullSort = retrieveTopkCpuFullSort(v_doc_copy, kNumToRetrieve, timeMsCpuFullSort1);

        if (v_topk_gpuBucketSort != v_topk_gpuFullSort)
        {
            throw std::runtime_error("Topk results from GPU bucket sort and GPU full sort do not match");
        }
        if (v_topk_gpuBucketSort != v_topk_cpuFullSort)
        {
            throw std::runtime_error("Topk results from GPU bucket sort and CPU full sort do not match");
        }

        if (t >= 0)
        {
            timeMsGpuBucketSort += timeMsGpuBucketSort1;
            timeMsGpuFullSort += timeMsGpuFullSort1;
            timeMsCpuFullSort += timeMsCpuFullSort1;
        }
    }

    timeMsGpuBucketSort /= kNumTrials;
    timeMsGpuFullSort /= kNumTrials;
    timeMsCpuFullSort /= kNumTrials;

    std::cout << "timeMsGpuBucketSort: " << timeMsGpuBucketSort << " ms" << std::endl;
    std::cout << "timeMsGpuFullSort: " << timeMsGpuFullSort << " ms" << std::endl;
    std::cout << "timeMsCpuFullSort: " << timeMsCpuFullSort << " ms" << std::endl;

    CHECK_CUDA(cudaFree(d_doc));
    CHECK_CUDA(cudaFree(d_buffer));
    retriever.reset();
}

int main()
{
    runExp(1000000);
    runExp(2000000);
    runExp(4000000);
    runExp(8000000);
    runExp(16000000);
    runExp(32000000);

    return 0;
}
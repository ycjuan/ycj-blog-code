#include <random>
#include <vector>
#include <stdexcept>
#include <iostream>

#include "common.cuh"
#include "topk.cuh"

using namespace std;

#define CHECK_CUDA(func)                                                                                                           \
    {                                                                                                                              \
        cudaError_t status = (func);                                                                                               \
        if (status != cudaSuccess)                                                                                                 \
        {                                                                                                                          \
            string error = "CUDA API failed at line " + to_string(__LINE__) + " with error: " + cudaGetErrorString(status) + "\n"; \
            throw runtime_error(error);                                                                                            \
        }                                                                                                                          \
    }

int kNumToRetrieve = 1000;
int kNumTrials = 10;

void runExp(int numDcos)
{
    default_random_engine generator;
    uniform_real_distribution<float> distribution(-1.0, 1.0);

    vector<Doc> v_doc(numDcos);
    for (int i = 0; i < numDcos; i++)
    {
        v_doc[i].docId = i;
        v_doc[i].score = distribution(generator);
    }

    TopkBucketSort retriever;
    Doc *d_doc = nullptr;
    Doc *d_buffer = nullptr;
    CHECK_CUDA(cudaMalloc(&d_doc, numDcos * sizeof(Doc)));
    CHECK_CUDA(cudaMalloc(&d_buffer, numDcos * sizeof(Doc)));

    double timeMsGpuBucketSort = 0;
    double timeMsGpuFullSort = 0;
    double timeMsCpuFullSort = 0;
    for (int t = -3; t < kNumTrials; t++)
    {
        float timeMsGpuBucketSort1 = 0;
        float timeMsGpuFullSort1 = 0;
        float timeMsCpuFullSort1 = 0;
        CHECK_CUDA(cudaMemcpy(d_doc, v_doc.data(), numDcos * sizeof(Doc), cudaMemcpyHostToDevice));
        vector<Doc> v_topk_gpuBucketSort = retriever.retrieveTopk(d_doc, d_buffer, numDcos, kNumToRetrieve, timeMsGpuBucketSort1);
        CHECK_CUDA(cudaMemcpy(d_doc, v_doc.data(), numDcos * sizeof(Doc), cudaMemcpyHostToDevice));
        vector<Doc> v_topk_gpuFullSort = retrieveTopkGpuFullSort(d_doc, numDcos, kNumToRetrieve, timeMsGpuFullSort1);
        vector<Doc> v_doc_copy = v_doc;
        vector<Doc> v_topk_cpuFullSort = retrieveTopkCpuFullSort(v_doc_copy, kNumToRetrieve, timeMsCpuFullSort1);

        if (v_topk_gpuBucketSort != v_topk_gpuFullSort)
        {
            throw runtime_error("Topk results from GPU bucket sort and GPU full sort do not match");
        }
        if (v_topk_gpuBucketSort != v_topk_cpuFullSort)
        {
            throw runtime_error("Topk results from GPU bucket sort and CPU full sort do not match");
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

    cout << "timeMsGpuBucketSort: " << timeMsGpuBucketSort << " ms" << endl;
    cout << "timeMsGpuFullSort: " << timeMsGpuFullSort << " ms" << endl;
    cout << "timeMsCpuFullSort: " << timeMsCpuFullSort << " ms" << endl;

    CHECK_CUDA(cudaFree(d_doc));
    CHECK_CUDA(cudaFree(d_buffer));
}

int main()
{
    runExp(10000);

    return 0;
}
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
            string error = "[main.cu] CUDA API failed at line " + to_string(__LINE__) + " with error: " + cudaGetErrorString(status) + "\n"; \
            throw runtime_error(error);                                                                                            \
        }                                                                                                                          \
    }

size_t kNumToRetrieve = 1000;
size_t kNumTrials = 10;

void runExp(size_t numReqs, size_t numDocs)
{
    cout << "\n\nrunning exps with numReq: " << numReqs << ", numDocs" << numDocs << endl;
    default_random_engine generator;
    uniform_real_distribution<float> distribution(-1.0, 1.0);

    float *dm_score = nullptr;
    CHECK_CUDA(cudaMallocManaged(&dm_score, numDocs * numReqs * sizeof(float)));
    for (size_t i = 0; i < numDocs * numReqs; i++)
    {
        dm_score[i] = distribution(generator);
    }

    double timeMsCpuFullSort = 0;
    double timeMsGpuFullSort = 0;
    double timeMsGpuSampling = 0;
    for (int t = -3; t < kNumTrials; t++)
    {
        float timeMsCpuFullSort1 = 0;
        float timeMsGpuFullSort1 = 0;
        float timeMsGpuSampling1 = 0;
        
        vector<Pair> v_topkCpuFullSort = retrieveTopkCpuFullSort(dm_score, numReqs, numDocs, kNumToRetrieve, timeMsGpuFullSort1);
        vector<Pair> v_topkGpuFullSort = retrieveTopkGpuFullSort(dm_score, numReqs, numDocs, kNumToRetrieve, timeMsGpuFullSort1);
        vector<Pair> v_topkGpuSampling;

        if (v_topkGpuFullSort != v_topkCpuFullSort)
        {
            throw runtime_error("Topk results from GPU full sort and CPU full sort do not match");
        }

        if (v_topkGpuSampling != v_topkCpuFullSort)
        {
            throw runtime_error("Topk results from GPU sampling and CPU full sort do not match");
        }

        if (t >= 0)
        {
            timeMsCpuFullSort += timeMsCpuFullSort1;
            timeMsGpuFullSort += timeMsGpuFullSort1;
            timeMsGpuSampling += timeMsGpuSampling1;
        }
    }

    timeMsCpuFullSort /= kNumTrials;
    timeMsGpuFullSort /= kNumTrials;
    timeMsGpuSampling /= kNumTrials;

    cout << "timeMsCpuFullSort: " << timeMsCpuFullSort << " ms" << endl;
    cout << "timeMsGpuFullSort: " << timeMsGpuFullSort << " ms" << endl;
    cout << "timeMsGpuSampling: " << timeMsGpuSampling << " ms" << endl;

}

int main()
{
    runExp(100, 1000000);

    return 0;
}
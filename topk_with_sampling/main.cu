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

int kNumToRetrieve = 100;
int kNumTrials = 100;

void runExp(int numReqs, int numDocs, bool useRandomSampling, bool useCpu)
{
    cout << "\n\nrunning exps with numReq: " << numReqs << ", numDocs: " << numDocs << endl;

    TopkParam param;
    param.numReqs = numReqs;
    param.numDocs = numDocs;
    param.numToRetrieve = kNumToRetrieve;
    param.useRandomSampling = useRandomSampling;

    size_t allocateInBytes;
    size_t totalAllocateInBytes = 0;
    
    allocateInBytes = (size_t)numDocs * numReqs * sizeof(float);
    CHECK_CUDA(cudaMallocManaged(&param.dm_score, allocateInBytes));
    totalAllocateInBytes += allocateInBytes;
    cout << "allocated " << allocateInBytes / 1024.0 / 1024.0 / 1024.0 << " GiB for dm_score" << endl;

    allocateInBytes = (size_t)numReqs * kNumToRetrieve * sizeof(Pair);
    CHECK_CUDA(cudaMallocHost(&param.hp_rstCpu, allocateInBytes));
    totalAllocateInBytes += allocateInBytes;
    cout << "allocated " << allocateInBytes / 1024.0 / 1024.0 / 1024.0 << " GiB for hp_rstCpu" << endl;

    allocateInBytes = (size_t)numReqs * kNumToRetrieve * sizeof(Pair);
    CHECK_CUDA(cudaMallocManaged(&param.dm_rstGpu, allocateInBytes));
    totalAllocateInBytes += allocateInBytes;
    cout << "allocated " << allocateInBytes / 1024.0 / 1024.0 / 1024.0 << " GiB for dm_rstGpu" << endl;

    cout << "total allocated " << totalAllocateInBytes / 1024.0 / 1024.0 / 1024.0 << " GiB" << endl;

    cout << "initializing scores" << endl;
    default_random_engine generator;
    uniform_real_distribution<float> distribution(-1.0, 1.0);
    for (size_t i = 0; i < numDocs * numReqs; i++)
    {
        param.dm_score[i] = distribution(generator);
    }
    cout << "scores initialized" << endl;

    TopkSampling topkSampling;
    topkSampling.malloc();

    double gpuSampleTimeMs = 0;
    double gpuFindThresholdTimeMs = 0;
    double gpuCopyEligibleTimeMs = 0;
    double gpuRetreiveExactTimeMs = 0;
    double gpuTotalTimeMs = 0;
    double gpuApproxTimeMs = 0;

    if (useCpu)
    {
        cout << "retrieving topk with cpu" << endl;
        retrieveTopkCpu(param);
        cout << "topk retrieved with cpu" << endl;
    }
    for (int t = -3; t < kNumTrials; t++)
    {
        topkSampling.retrieveTopk(param);

        if (useCpu && t == -1)
        {
            cout << "compare results" << endl;
            for (int reqIdx = 0; reqIdx < numReqs; reqIdx++)
            {
                for (int docIdx = 0; docIdx < kNumToRetrieve; docIdx++)
                {
                    size_t memAddr = getMemAddr(reqIdx, docIdx, kNumToRetrieve);
                    Pair cpuPair = param.hp_rstCpu[memAddr];
                    Pair gpuPair = param.dm_rstGpu[memAddr];
                    if (cpuPair.docIdx != gpuPair.docIdx || cpuPair.score != gpuPair.score)
                    {
                        cout << "mismatch at reqIdx: " << reqIdx << ", docIdx: " << docIdx << endl;
                        cout << "cpuPair: " << cpuPair.reqIdx << ", " << cpuPair.docIdx << ", " << cpuPair.score << endl;
                        cout << "gpuPair: " << gpuPair.reqIdx << ", " << gpuPair.docIdx << ", " << gpuPair.score << endl;
                    }
                }
            }
            cout << "results compared" << endl;
        }

        if (t >= 0)
        {
            gpuSampleTimeMs += param.gpuSampleTimeMs;
            gpuFindThresholdTimeMs += param.gpuFindThresholdTimeMs;
            gpuCopyEligibleTimeMs += param.gpuCopyEligibleTimeMs;
            gpuRetreiveExactTimeMs += param.gpuRetreiveExactTimeMs;
            gpuTotalTimeMs += param.gpuTotalTimeMs;
            gpuApproxTimeMs += param.gpuApproxTimeMs;
        }
    }

    cout << "gpuSampleTimeMs: " << gpuSampleTimeMs / kNumTrials << endl;
    cout << "gpuFindThresholdTimeMs: " << gpuFindThresholdTimeMs / kNumTrials << endl;
    cout << "gpuCopyEligibleTimeMs: " << gpuCopyEligibleTimeMs / kNumTrials << endl;
    cout << "gpuRetreiveExactTimeMs: " << gpuRetreiveExactTimeMs / kNumTrials << endl;
    cout << "gpuTotalTimeMs: " << gpuTotalTimeMs / kNumTrials << endl;
    cout << "gpuApproxTimeMs: " << gpuApproxTimeMs / kNumTrials << endl;

    CHECK_CUDA(cudaFree(param.dm_score));
    CHECK_CUDA(cudaFreeHost(param.hp_rstCpu));
    CHECK_CUDA(cudaFree(param.dm_rstGpu));
}

int main()
{
    runExp(64, 1600000, true, true);

    return 0;
}
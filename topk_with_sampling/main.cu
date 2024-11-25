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
int kNumTrials = 10;

void runExp(int numReqs, int numDocs)
{
    cout << "\n\nrunning exps with numReq: " << numReqs << ", numDocs: " << numDocs << endl;

    TopkParam param;
    param.numReqs = numReqs;
    param.numDocs = numDocs;
    param.numToRetrieve = kNumToRetrieve;

    CHECK_CUDA(cudaMallocManaged(&param.dm_score, numDocs * numReqs * sizeof(float)));
    CHECK_CUDA(cudaMallocHost(&param.hp_rstCpu, numReqs * kNumToRetrieve * sizeof(Pair)));
    CHECK_CUDA(cudaMallocManaged(&param.dm_rstGpu, numReqs * kNumToRetrieve * sizeof(Pair)));

    default_random_engine generator;
    uniform_real_distribution<float> distribution(-1.0, 1.0);
    for (size_t i = 0; i < numDocs * numReqs; i++)
    {
        param.dm_score[i] = distribution(generator);
    }

    TopkSampling topkSampling;
    topkSampling.malloc();

    double gpuSampleTimeMs = 0;
    double gpuFindThresholdTimeMs = 0;
    double gpuCopyEligibleTimeMs = 0;
    double gpuRetreiveExactTimeMs = 0;
    double gpuTotalTimeMs = 0;
    double gpuApproxTimeMs = 0;

    retrieveTopkCpu(param);
    for (int t = -3; t < kNumTrials; t++)
    {
        topkSampling.retrieveTopk(param);

        if (t == -1)
        {
            for (int reqIdx = 0; reqIdx < numReqs; reqIdx++)
            {
                for (int docIdx = 0; docIdx < kNumToRetrieve; docIdx++)
                {
                    size_t memAddr = getMemAddr(reqIdx, docIdx, kNumToRetrieve);
                    Pair cpuPair = param.hp_rstCpu[memAddr];
                    Pair gpuPair = param.dm_rstGpu[memAddr];
                    if (cpuPair.docId != gpuPair.docId || cpuPair.score != gpuPair.score)
                    {
                        cout << "mismatch at reqIdx: " << reqIdx << ", docIdx: " << docIdx << endl;
                        cout << "cpuPair: " << cpuPair.reqId << ", " << cpuPair.docId << ", " << cpuPair.score << endl;
                        cout << "gpuPair: " << gpuPair.reqId << ", " << gpuPair.docId << ", " << gpuPair.score << endl;
                        break;
                    }
                }
            }
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
    runExp(2, 16000000);

    return 0;
}
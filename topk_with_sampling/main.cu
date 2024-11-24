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
size_t kNumTrials = 1;

void runExp(size_t numReqs, size_t numDocs)
{
    cout << "\n\nrunning exps with numReq: " << numReqs << ", numDocs" << numDocs << endl;
    default_random_engine generator;
    uniform_real_distribution<float> distribution(-1.0, 1.0);


    TopkParam param;
    param.numReqs = numReqs;
    param.numDocs = numDocs;
    param.numToRetrieve = kNumToRetrieve;

    CHECK_CUDA(cudaMallocManaged(&param.dm_score, numDocs * numReqs * sizeof(float)));
    CHECK_CUDA(cudaMallocHost(&param.hp_rstCpu, numReqs * kNumToRetrieve * sizeof(Pair)));
    CHECK_CUDA(cudaMallocManaged(&param.dm_rstGpu, numReqs * kNumToRetrieve * sizeof(Pair)));

    for (size_t i = 0; i < numDocs * numReqs; i++)
    {
        param.dm_score[i] = distribution(generator);
    }

    TopkSampling topkSampling;
    topkSampling.malloc();

    for (int t = -3; t < kNumTrials; t++)
    {
        retrieveTopkCpu(param);
        topkSampling.retrieveTopk(param);

        for (int reqIdx = 0; reqIdx < numReqs; reqIdx++)
        {
            for (int docIdx = 0; docIdx < kNumToRetrieve; docIdx++)
            {
                size_t memAddr = getMemAddr(reqIdx, docIdx, kNumToRetrieve);
                Pair cpuPair = param.hp_rstCpu[memAddr];
                Pair gpuPair = param.dm_rstGpu[memAddr];
                if (cpuPair.docId != gpuPair.docId || cpuPair.score != gpuPair.score)
                {
                    cout << "Error: CPU and GPU results do not match" << endl;
                    break;
                }
            }
        }

        for (int i = 0; i < numReqs * kNumToRetrieve; i++)
        {
            Pair cpuPair = param.hp_rstCpu[i];
            Pair gpuPair = param.dm_rstGpu[i];
            if (param.hp_rstCpu[i].docId != param.dm_rstGpu[i].docId)
            {
                cout << "Error: CPU and GPU results do not match" << endl;
                break;
            }
        }
    }

    CHECK_CUDA(cudaFree(param.dm_score));
    CHECK_CUDA(cudaFree(param.hp_rstCpu));
    CHECK_CUDA(cudaFree(param.dm_rstGpu));
}

int main()
{
    runExp(100, 1000000);

    return 0;
}
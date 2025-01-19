#include <iostream>
#include <vector>

#include "emb.cuh"

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

void checkRst(EmbData data)
{
    vector<Pair> v_rstGpu(data.maskSize);
    CHECK_CUDA(cudaMemcpy(v_rstGpu.data(), data.d_mask, data.maskSize * sizeof(Pair), cudaMemcpyDeviceToHost));
    for (size_t maskIdx = 0; maskIdx < data.maskSize; maskIdx++)
    {
            Pair cpuPair = data.h_mask[maskIdx];
            Pair gpuPair = v_rstGpu[maskIdx];
            if (cpuPair.docIdx != gpuPair.docIdx || abs(cpuPair.score - gpuPair.score) / abs(cpuPair.score) > 1e-3)
            {
                cout << "mismatch at maskIdx: " << maskIdx << endl;
                cout << "cpuPair: " << cpuPair.reqIdx << ", " << cpuPair.docIdx << ", " << cpuPair.score << endl;
                cout << "gpuPair: " << gpuPair.reqIdx << ", " << gpuPair.docIdx << ", " << gpuPair.score << endl;
            }
        }
    cout << "results compared!!!" << endl;
}

int main()
{
    int kNumDocs = 1000000;
    int kNumReqs = 16;
    int kEmbDim = 128;
    int kTrials = 100;
    float kPassRate = 0.1;
    bool kSkipVerification = false;

    EmbData data;
    cout << "Initializing data..." << endl;
    data.initRand(kNumDocs, kNumReqs, kEmbDim, ROW_MAJOR, ROW_MAJOR);
    data.initRandMask(kPassRate);
    // Note. It looks like cublasGemmEx did some shape checking, so every combination of mem layouts have similar performance.

    cout << "Running embMaskOpGpu..." << endl;
    double timeMsCuBlasSum = 0;
    for (int i = -3; i < kTrials; i++)
    {
        embMaskOpGpu(data);
        if (i >= 0)
            timeMsCuBlasSum += data.timeMsCuBlas;
    }
    cout << "Average time for embMaskOpGpu: " << timeMsCuBlasSum / kTrials << " ms" << endl;

    if (!kSkipVerification)
    {
        cout << "Running embMaskOpCpu..." << endl;
        embMaskOpCpu(data);

        cout << "Checking results..." << endl;
        checkRst(data);
    }

    return 0;
}
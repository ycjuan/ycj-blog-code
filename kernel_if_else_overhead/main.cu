#include <iostream>

#include "util.cuh"
#include "kernel.cuh"

using namespace std;

int kDataSize = 1 << 20;
int kNumCountInc = 1 << 10;
int kNumTrials = 100;

#define CHECK_CUDA(func)                                                                                                           \
    {                                                                                                                              \
        cudaError_t status = (func);                                                                                               \
        if (status != cudaSuccess)                                                                                                 \
        {                                                                                                                          \
            string error = "CUDA API failed at line " + to_string(__LINE__) + " with error: " + cudaGetErrorString(status) + "\n"; \
            throw runtime_error(error);                                                                                            \
        }                                                                                                                          \
    }

int main()
{
    Param param;
    param.dataSize = kDataSize;
    param.numCountInc = kNumCountInc;
    param.numTrials = kNumTrials;
    CHECK_CUDA(cudaMalloc(&param.d_count, kDataSize * sizeof(long)));

    param.method = METHOD0;
    runExp(param);

    param.method = METHOD1;
    runExp(param);

    param.method = METHOD2;
    runExp(param);

    param.method = METHOD3;
    runExp(param);

    return 0;
}

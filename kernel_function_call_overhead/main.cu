#include <iostream>

#include "util.cuh"
#include "methods.cuh"

using namespace std;

int kDataSize = 1 << 10;
int kNumCountInc = 1 << 5;
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

    runSetupBaseline(param);

    runSetupA(param);

    return 0;
}

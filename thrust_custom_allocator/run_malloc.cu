#include <vector>
#include <thread>
#include <iostream>
#include <random>
#include <thrust/execution_policy.h>
#include <thrust/sort.h>

#include "util.cuh"

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

int main()
{
    int kNumElements = 1000000;
    float *d_data;

    for (int i = 0; ; i++)
    {
        CudaTimer timer;

        timer.tic();
        CHECK_CUDA(cudaMalloc(&d_data, kNumElements * sizeof(float)))
        cudaFree(d_data);
        float timeMs = timer.tocMs();

        if (timeMs > 50 ||  i % 100 == 0)
            cout << "i = " << i << ", timeMs = " << timeMs << "ms" << endl;

        this_thread::sleep_for(chrono::milliseconds(10));
    }

    return 0;
}
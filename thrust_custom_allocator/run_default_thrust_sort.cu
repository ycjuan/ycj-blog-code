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
    CHECK_CUDA(cudaMallocManaged(&d_data, kNumElements * sizeof(float)))
    std::default_random_engine gen;
    std::uniform_real_distribution<float> floatDist(-1.0, 1.0);
    for (int i = 0; i < kNumElements; i++)
        d_data[i] = floatDist(gen);

    for (int i = 0; ; i++)
    {
        CudaTimer timer;

        timer.tic();
        thrust::stable_sort(thrust::device, d_data, d_data + kNumElements);
        float timeMs = timer.tocMs();

        if (timeMs > 50 ||  i % 100 == 0)
            cout << "i = " << i << ", timeMs = " << timeMs << "ms" << endl;

        this_thread::sleep_for(chrono::milliseconds(10));
    }

    cudaFree(d_data);
    return 0;
}
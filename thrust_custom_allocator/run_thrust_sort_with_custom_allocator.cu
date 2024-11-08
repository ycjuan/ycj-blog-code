#include <vector>
#include <thread>
#include <iostream>
#include <sstream>
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

// Modified from https://github.com/NVIDIA/thrust/blob/main/examples/cuda/custom_temporary_allocation.cu
struct ThrustAllocator
{
    typedef char value_type;

    void malloc(size_t numBytes)
    {
        cudaError_t cudaError = cudaMalloc(&d_arr, numBytes);
        if (cudaError != cudaSuccess)
        {
            ostringstream errorMsg;
            errorMsg << "[ThrustAllocator::malloc] cudaMalloc error: " << cudaGetErrorString(cudaError)
                     << ", numBytes = " << numBytes;
            throw std::runtime_error(errorMsg.str());
        }
        numBytesAllocated = numBytes;
    }

    void free()
    {
        if (d_arr != nullptr)
        {
            cudaFree(d_arr);
            d_arr = nullptr;
            numBytesAllocated = 0;
        }
    }

    char *allocate(std::ptrdiff_t numBytesAsked)
    {
        if (verbose)
            cout << "[ThrustAllocator::allocate]: numBytesAsked = " << numBytesAsked << ", numBytesAllocated = " << numBytesAllocated;

        if (numBytesAsked > numBytesAllocated)
        {
            std::ostringstream errorMsg;
            errorMsg << "[ThrustAllocator::allocate] numBytesAsked > numBytesAllocated: "
                     << "numBytesAsked = " << numBytesAsked
                     << ", numBytesAllocated = " << numBytesAllocated;
            throw std::runtime_error(errorMsg.str());
        }

        return d_arr;
    }

    void deallocate(char *ptr, size_t)
    {
        // Do nothing
    }

    bool verbose = false;

    char *d_arr = nullptr;

    size_t numBytesAllocated = 0;
};

int main()
{
    int kNumElements = 1000000;
    float *d_data;
    CHECK_CUDA(cudaMallocManaged(&d_data, kNumElements * sizeof(float)))
    std::default_random_engine gen;
    std::uniform_real_distribution<float> floatDist(-1.0, 1.0);
    for (int i = 0; i < kNumElements; i++)
        d_data[i] = floatDist(gen);

    ThrustAllocator allocator;
    allocator.malloc(kNumElements * sizeof(float) + 1000000);

    for (int i = 0;; i++)
    {
        allocator.verbose = i % 100 == 0;

        CudaTimer timer;

        timer.tic();
        thrust::sort(thrust::cuda::par(allocator), d_data, d_data + kNumElements);
        float timeMs = timer.tocMs();

        if (timeMs > 50 || i % 100 == 0)
            cout << "i = " << i << ", timeMs = " << timeMs << "ms" << endl;

        this_thread::sleep_for(chrono::milliseconds(10));
    }

    cudaFree(d_data);
    return 0;
}
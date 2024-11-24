#ifndef THRUST_ALLOCATOR_CUH
#define THRUST_ALLOCATOR_CUH

#include <vector>
#include <thread>
#include <iostream>
#include <sstream>
#include <random>
#include <thrust/execution_policy.h>
#include <thrust/sort.h>

using namespace std;

// Modified from https://github.com/NVIDIA/thrust/blob/main/examples/cuda/custom_temporary_allocation.cu
struct StaticThrustAllocator
{
    typedef char value_type;

    void malloc(size_t numBytes)
    {
        cudaError_t cudaError = cudaMalloc(&d_arr, numBytes);
        if (cudaError != cudaSuccess)
        {
            ostringstream errorMsg;
            errorMsg << "[StaticThrustAllocator::malloc] cudaMalloc error: " << cudaGetErrorString(cudaError)
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
            cout << "[StaticThrustAllocator::allocate]: numBytesAsked = " << numBytesAsked << ", numBytesAllocated = " << numBytesAllocated << endl;

        if (numBytesAsked > numBytesAllocated)
        {
            std::ostringstream errorMsg;
            errorMsg << "[StaticThrustAllocator::allocate] numBytesAsked > numBytesAllocated: "
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

#endif // THRUST_ALLOCATOR_CUH
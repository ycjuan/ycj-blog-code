#include <cassert>
#include "cuda_malloc_raii.cuh"

size_t getFreeMemoryInBytes()
{
    size_t freeMemoryInBytes;
    cudaError_t cudaError = cudaMemGetInfo(&freeMemoryInBytes, nullptr);
    if (cudaError != cudaSuccess)
    {
        throw std::runtime_error("Failed to get free memory: " + std::string(cudaGetErrorString(cudaError)));
    }
    return freeMemoryInBytes;
}

int main()
{
    // --------------
    // Some constants
    const size_t kArraySize = 1000000;

    // --------------
    // Test CudaDeviceArray
    {
        size_t freeMemBeforeMalloc = getFreeMemoryInBytes();
        {
            CudaDeviceArray<float> arr(kArraySize, "arr1");
            assert(arr.data() != nullptr);
            assert(arr.getArraySize() == kArraySize);
            assert(arr.getElementSize() == sizeof(float));
            assert(arr.getArraySizeInBytes() == kArraySize * sizeof(float));
            assert(arr.getName() == "arr1");
        }
        size_t freeMemAfterFree = getFreeMemoryInBytes();
        assert(freeMemBeforeMalloc == freeMemAfterFree);
    }

    // --------------
    // Test CudaHostArray
    {
        // It's a bit hard to use the free memory to assert. So we will just eye-ball the logs.
        CudaHostArray<float> arr(kArraySize, "arr1");
    }

    // --------------
    // Test Polymorphism
    {
        size_t freeMemBeforeMalloc = getFreeMemoryInBytes();
        CudaArray<float> *arr = new CudaDeviceArray<float>(kArraySize, "arr1");
        delete arr;
        size_t freeMemAfterFree = getFreeMemoryInBytes();
        assert(freeMemBeforeMalloc == freeMemAfterFree); // Make sure the destructor of the derived class is called
    }

    return 0;
}
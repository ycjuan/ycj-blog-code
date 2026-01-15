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
        std::cout << "======== Test CudaDeviceArray ========" << std::endl;
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
        std::cout << "======== Test CudaHostArray ========" << std::endl;
        // It's a bit hard to use the free memory to assert. So we will just eye-ball the logs.
        CudaHostArray<float> arr(kArraySize, "arr1");
    }

    // --------------
    // Test Polymorphism
    {
        std::cout << "======== Test Polymorphism ========" << std::endl;
        size_t freeMemBeforeMalloc = getFreeMemoryInBytes();
        CudaArray<float> *arr = new CudaDeviceArray<float>(kArraySize, "arr1");
        delete arr;
        size_t freeMemAfterFree = getFreeMemoryInBytes();
        assert(freeMemBeforeMalloc == freeMemAfterFree); // Make sure the destructor of the derived class is called
    }

    // --------------
    // Test copy
    {
        std::cout << "======== Test copy ========" << std::endl;
        size_t freeMemBeforeMalloc = getFreeMemoryInBytes();
        {
            CudaDeviceArray<float> arr1(kArraySize, "arr1");
            {
                auto arr2 = arr1;
            }
            size_t freeMemAfterArr2Free = getFreeMemoryInBytes();
            assert(freeMemAfterArr2Free < freeMemBeforeMalloc); // arr2 is out of scope, but arr1 is still in scope, so the memory should not be freed
        }
        size_t freeMemAfterArr1Free = getFreeMemoryInBytes();
        assert(freeMemAfterArr1Free == freeMemBeforeMalloc); // arr1 is out of scope too, so the memory should be freed
    }

    return 0;
}
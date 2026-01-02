#pragma once

#include <chrono>
#include <iostream>
#include <stdexcept>

#define CHECK_CUDA(func)                                                                                               \
    {                                                                                                                  \
        cudaError_t status = (func);                                                                                   \
        if (status != cudaSuccess)                                                                                     \
        {                                                                                                              \
            std::string error = "[util.cuh] CUDA API failed at line " + std::to_string(__LINE__)                       \
                + " with error: " + cudaGetErrorString(status) + "\n";                                                 \
            throw std::runtime_error(error);                                                                           \
        }                                                                                                              \
    }

#define CHECK_CUBLAS(func)                                                                                             \
    {                                                                                                                  \
        cublasStatus_t status = (func);                                                                                \
        if (status != CUBLAS_STATUS_SUCCESS)                                                                           \
        {                                                                                                              \
            std::string error = "[util.cuh] CUBLAS API failed at line " + std::to_string(__LINE__)                     \
                + " with error: " + std::to_string(status) + "\n";                                               \
            throw std::runtime_error(error);                                                                           \
        }                                                                                                              \
    }

class Timer
{
public:
    void tic() { start_ = std::chrono::high_resolution_clock::now(); }

    float tocMs()
    {
        auto stop = std::chrono::high_resolution_clock::now();
        std::chrono::microseconds duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start_);
        float timeMs = duration.count() / 1000.0;
        return timeMs;
    }

private:
    std::chrono::high_resolution_clock::time_point start_;
};

inline void printDeviceInfo()
{
    using namespace std;

    int deviceCount;
    cudaError_t cudaError = cudaGetDeviceCount(&deviceCount);
    if (cudaError != cudaSuccess)
    {
        throw std::runtime_error("Error getting device count: " + std::string(cudaGetErrorString(cudaError)));
    }
    for (int i = 0; i < deviceCount; ++i)
    {
        cudaDeviceProp prop;
        cudaError = cudaGetDeviceProperties(&prop, i);
        if (cudaError != cudaSuccess)
        {
            throw std::runtime_error("Error getting device properties: " + std::string(cudaGetErrorString(cudaError)));
        }
        cout << "Device " << i << ": " << prop.name << endl;
        cout << "  Total global memory: " << prop.totalGlobalMem / (1024 * 1024) << " MB" << endl;
        cout << "  Multiprocessor count: " << prop.multiProcessorCount << endl;
        cout << "  Max threads per block: " << prop.maxThreadsPerBlock << endl;
        cout << "  Max threads dim: (" << prop.maxThreadsDim[0] << ", " << prop.maxThreadsDim[1] << ", "
             << prop.maxThreadsDim[2] << ")" << endl;
        cout << "  Max grid size: (" << prop.maxGridSize[0] << ", " << prop.maxGridSize[1] << ", "
             << prop.maxGridSize[2] << ")" << endl;
        cout << "  Clock rate: " << prop.clockRate / 1000 << " MHz" << endl;
        cout << "  Compute capability: " << prop.major << "." << prop.minor << endl;
        cout << "  Memory clock rate: " << prop.memoryClockRate / 1000 << " MHz" << endl;
        cout << "  Memory bus width: " << prop.memoryBusWidth << " bits" << endl;
        cout << "  L2 cache size: " << prop.l2CacheSize / 1024 << " KB" << endl;
        cout << "  concurrentKernels: " << (prop.concurrentKernels ? "Yes" : "No") << endl;
        cout << "  asyncEngineCount: " << prop.asyncEngineCount << endl;
        cout << "  streamPrioritiesSupported: " << (prop.streamPrioritiesSupported ? "Yes" : "No") << endl;
        cout << "  unifiedAddressing: " << (prop.unifiedAddressing ? "Yes" : "No") << endl;
        cout << "  pageableMemoryAccess: " << (prop.pageableMemoryAccess ? "Yes" : "No") << endl;
        cout << "  canMapHostMemory: " << (prop.canMapHostMemory ? "Yes" : "No") << endl;
        cout << "  sharedMemPerBlock: " << prop.sharedMemPerBlock / 1024 << " KB" << endl;
        cout << "  sharedMemPerMultiprocessor: " << prop.sharedMemPerMultiprocessor / 1024 << " KB" << endl;
    }
}
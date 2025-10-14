#pragma once

#include <chrono>
#include <stdexcept>
#include <iostream>

namespace MatMatMulFromScratch {

#define CHECK_CUDA(call)                                                                                                        \
    do {                                                                                                                        \
        cudaError_t status = call;                                                                                              \
        if (status != cudaSuccess) {                                                                                            \
            throw std::runtime_error(                                                                                           \
                std::string("CUDA error at ") + __FILE__ + ":" + std::to_string(__LINE__) + ": " + cudaGetErrorString(status)); \
        }                                                                                                                       \
    } while (0)

class CudaTimer
{
public:
    CudaTimer()
    {
        cudaEventCreate(&start_);
        cudaEventCreate(&stop_);
    }

    void tic()
    {
        cudaEventRecord(start_);
    }

    float tocMs()
    {
        cudaEventRecord(stop_);
        cudaEventSynchronize(stop_);
        float elapsedMs;
        cudaEventElapsedTime(&elapsedMs, start_, stop_);
        return elapsedMs;
    }

    ~CudaTimer()
    {
        cudaEventDestroy(start_);
        cudaEventDestroy(stop_);
    }

private:
    cudaEvent_t start_;
    cudaEvent_t stop_;
};


class Timer
{
public:

    void tic()
    {
        start_ = std::chrono::high_resolution_clock::now();
    }

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
        cout << "  Max threads dim: (" << prop.maxThreadsDim[0] << ", " << prop.maxThreadsDim[1] << ", " << prop.maxThreadsDim[2] << ")" << endl;
        cout << "  Max grid size: (" << prop.maxGridSize[0] << ", " << prop.maxGridSize[1] << ", " << prop.maxGridSize[2] << ")" << endl;
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

} // namespace BatchScalibility
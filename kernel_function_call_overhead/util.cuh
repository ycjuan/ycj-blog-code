#ifndef UTIL_CUH
#define UTIL_CUH

#include <chrono>

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

#endif // UTIL_CUH
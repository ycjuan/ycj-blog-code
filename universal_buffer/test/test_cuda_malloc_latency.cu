#include <chrono>
#include <iostream>
#include <thread>
#include <cuda_runtime.h>

using Clock = std::chrono::high_resolution_clock;
using Ms    = std::chrono::duration<double, std::milli>;

int main()
{
    const int    kTrials        = 10000;
    const int    kPrintInterval = 1000;
    const double kSpikeThreshMs = 10.0;
    const size_t kAllocBytes    = 1024 * 1024; // 1 MB

    double totalMallocMs = 0.0;
    double totalFreeMs   = 0.0;
    int    spikes        = 0;

    for (int i = 1; i <= kTrials; ++i)
    {
        void* ptr = nullptr;

        auto t0 = Clock::now();
        cudaMalloc(&ptr, kAllocBytes);
        double mallocMs = Ms(Clock::now() - t0).count();

        std::this_thread::sleep_for(std::chrono::milliseconds(1));

        auto t1 = Clock::now();
        cudaFree(ptr);
        double freeMs = Ms(Clock::now() - t1).count();

        totalMallocMs += mallocMs;
        totalFreeMs   += freeMs;

        if (mallocMs > kSpikeThreshMs || freeMs > kSpikeThreshMs)
        {
            std::cout << "[SPIKE] trial " << i
                      << " malloc=" << mallocMs << "ms"
                      << " free="   << freeMs   << "ms" << std::endl;
            ++spikes;
        }

        if (i % kPrintInterval == 0)
        {
            std::cout << "trial " << i
                      << " avg_malloc=" << totalMallocMs / i << "ms"
                      << " avg_free="   << totalFreeMs   / i << "ms"
                      << " spikes="     << spikes << std::endl;
        }
    }

    return 0;
}

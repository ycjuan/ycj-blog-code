// Real CUDA D2H (device-to-host) copy implementation, compiled with nvcc.
// Allocates device memory with cudaMalloc, uploads the result vector once
// (untimed), then times repeated cudaMemcpy(..., cudaMemcpyDeviceToHost)
// calls -- an actual GPU->CPU transfer, not a same-host analogy.

#include "cuda_copy.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>

#include <cuda_runtime.h>

using Clock = std::chrono::high_resolution_clock;

namespace
{
void checkCuda(cudaError_t err, const char* what)
{
    if (err != cudaSuccess)
    {
        std::fprintf(stderr,
                     "CUDA error during %s: %s\n"
                     "(This benchmark's GPU copy path requires a CUDA-capable GPU at "
                     "runtime; run without the GPU flag, or on a GPU-enabled node.)\n",
                     what,
                     cudaGetErrorString(err));
        std::exit(1);
    }
}

double percentileUs(std::vector<double> latenciesUs, double pct)
{
    if (latenciesUs.empty())
    {
        return 0.0;
    }
    std::sort(latenciesUs.begin(), latenciesUs.end());
    double rank = (pct / 100.0) * (latenciesUs.size() - 1);
    size_t lo   = static_cast<size_t>(rank);
    size_t hi   = std::min(lo + 1, latenciesUs.size() - 1);
    double frac = rank - lo;
    return latenciesUs[lo] + frac * (latenciesUs[hi] - latenciesUs[lo]);
}
} // namespace

GpuCopyStats timedCudaD2HCopy(const std::vector<int>& src, int reps)
{
    int*   d_result = nullptr;
    size_t numBytes = src.size() * sizeof(int);

    // Guard against cudaMalloc(0): still valid, but keep behavior explicit
    // for empty results.
    checkCuda(cudaMalloc(reinterpret_cast<void**>(&d_result), numBytes == 0 ? sizeof(int) : numBytes), "cudaMalloc");

    if (!src.empty())
    {
        checkCuda(cudaMemcpy(d_result, src.data(), numBytes, cudaMemcpyHostToDevice),
                  "cudaMemcpy H2D (initial upload, untimed)");
    }

    std::vector<double> latenciesUs;
    latenciesUs.reserve(reps);
    std::vector<int> hostResult(src.size());
    for (int r = 0; r < reps; r++)
    {
        auto start = Clock::now();
        if (!src.empty())
        {
            checkCuda(cudaMemcpy(hostResult.data(), d_result, numBytes, cudaMemcpyDeviceToHost), "cudaMemcpy D2H");
        }
        auto end = Clock::now();
        latenciesUs.push_back(std::chrono::duration<double, std::micro>(end - start).count());
    }

    checkCuda(cudaFree(d_result), "cudaFree");

    GpuCopyStats stats;
    stats.p50Us = percentileUs(latenciesUs, 50.0);
    stats.p95Us = percentileUs(latenciesUs, 95.0);
    return stats;
}

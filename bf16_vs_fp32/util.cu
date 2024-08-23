#include "util.cuh"

CudaTimer::CudaTimer()
{
    cudaEventCreate(&start_);
    cudaEventCreate(&stop_);
}

void CudaTimer::tic()
{
    cudaEventRecord(start_);
}

float CudaTimer::tocMs()
{
    cudaEventRecord(stop_);
    cudaEventSynchronize(stop_);
    float elapsedMs;
    cudaEventElapsedTime(&elapsedMs, start_, stop_);
    return elapsedMs;
}

CudaTimer::~CudaTimer()
{
    cudaEventDestroy(start_);
    cudaEventDestroy(stop_);
}

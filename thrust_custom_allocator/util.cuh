#ifndef UTIL_CUH
#define UTIL_CUH

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

#endif // UTIL_CUH
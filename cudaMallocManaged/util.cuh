#ifndef UTIL_CUH
#define UTIL_CUH

class CudaTimer
{
public:
    CudaTimer();
    void tic();
    float tocMs();
    ~CudaTimer();

private:
    cudaEvent_t start_;
    cudaEvent_t stop_;
};

size_t getCpuRamUsageByte();

float getCpuRamUsageMiB();

#endif // UTIL_CUH
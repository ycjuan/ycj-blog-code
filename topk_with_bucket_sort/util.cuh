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

#endif // UTIL_CUH
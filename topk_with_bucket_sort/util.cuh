#pragma once

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
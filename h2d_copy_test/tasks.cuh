#pragma once

#include <string>
#include <cuda_bf16.h>

class BaseRunner
{
public:
    BaseRunner(uint64_t m, uint64_t n, uint64_t k);
    ~BaseRunner();
    virtual void run() = 0;
    virtual std::string getName() = 0;

protected:
    uint64_t m_m;
    uint64_t m_n;
    uint64_t m_k;
    __nv_bfloat16* m_d_A;
    __nv_bfloat16* m_d_B;
    float* m_d_C;
    cudaStream_t m_stream;
};

class CudaCoreMatMatMulRunner : public BaseRunner
{
public:
    using BaseRunner::BaseRunner;
    void run() override;
    std::string getName() override { return "CudaCoreMatMatMul"; }
};

class TensorCoreMatMatMulRunner : public BaseRunner
{
public:
    using BaseRunner::BaseRunner;
    void run() override;
    std::string getName() override { return "TensorCoreMatMatMul"; }
};

class H2DMemcpyRunner : public BaseRunner
{
public:
    H2DMemcpyRunner(int m, int n, int k);
    void run() override;
    std::string getName() override { return "H2DMemcpy"; }

private:
    __nv_bfloat16* m_h_A;
};
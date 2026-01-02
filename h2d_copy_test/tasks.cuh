#pragma once

#include <cuda_bf16.h>

class BaseRunner
{
public:
    BaseRunner(uint64_t m, uint64_t n, uint64_t k);
    ~BaseRunner();
    virtual void run() = 0;

protected:
    uint64_t m_m;
    uint64_t m_n;
    uint64_t m_k;
    __nv_bfloat16* m_d_A;
    __nv_bfloat16* m_d_B;
    float* m_d_C;
};

class CudaCoreMatMatMulRunner : public BaseRunner
{
public:
    using BaseRunner::BaseRunner;
    void run() override;
};

class TensorCoreMatMatMulRunner : public BaseRunner
{
public:
    using BaseRunner::BaseRunner;
    void run() override;
};

class H2DMemcpyRunner : public BaseRunner
{
public:
    H2DMemcpyRunner(int m, int n, int k);
    void run() override;

private:
    __nv_bfloat16* m_h_A;
};
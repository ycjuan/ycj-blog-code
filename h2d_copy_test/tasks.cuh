#pragma once

class MatMatMulRunner
{
public:
    MatMatMulRunner(int m, int n, int k);
    ~MatMatMulRunner();
    virtual void run() = 0;

private:
    int m_;
    int n_;
    int k_;
    float* d_A_;
    float* d_B_;
    float* d_C_;
};

class CudaCoreMatMatMulRunner : public MatMatMulRunner
{
public:
    using MatMatMulRunner::MatMatMulRunner;
    //~CudaCoreMatMatMatMulRunner();
    void run() override;
};

class TensorCoreMatMatMulRunner : public MatMatMulRunner
{
public:
    using MatMatMulRunner::MatMatMulRunner;
    //~TensorCoreMatMatMulRunner();
    void run() override;
};
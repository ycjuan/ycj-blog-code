#include <iostream>
#include "tasks.cuh"
#include "util.cuh"

MatMatMulRunner::MatMatMulRunner(int m, int n, int k)
    : m_(m), n_(n), k_(k)
{
    std::cout << "MatMatMulRunner::MatMatMulRunner(" << m_ << ", " << n_ << ", " << k_ << ")" << std::endl;
    //cudaMalloc(&d_A_, m_ * k_ * sizeof(float));
    //cudaMalloc(&d_B_, k_ * n_ * sizeof(float));
    //cudaMalloc(&d_C_, m_ * n_ * sizeof(float));
}

MatMatMulRunner::~MatMatMulRunner()
{
    std::cout << "MatMatMulRunner::~MatMatMulRunner()" << std::endl;
    //cudaFree(d_A_);
    //cudaFree(d_B_);
    //cudaFree(d_C_);
}

void CudaCoreMatMatMulRunner::run()
{
    std::cout << "CudaCoreMatMatMulRunner::run()" << std::endl;
}

void TensorCoreMatMatMulRunner::run()
{
    std::cout << "TensorCoreMatMatMulRunner::run()" << std::endl;
}
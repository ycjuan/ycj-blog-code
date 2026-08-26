#pragma once

#include <chrono>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <iostream>
#include <stdexcept>
#include <string>

#define CHECK_CUDA(func)                                                                                               \
    {                                                                                                                  \
        cudaError_t status = (func);                                                                                   \
        if (status != cudaSuccess)                                                                                     \
        {                                                                                                              \
            std::string error = "[partition_sharding_scaling] CUDA API failed at " + std::string(__FILE__) + ":"       \
                                + std::to_string(__LINE__) + " with error: " + cudaGetErrorString(status) + "\n";      \
            throw std::runtime_error(error);                                                                           \
        }                                                                                                              \
    }

#define CHECK_CUBLAS(func)                                                                                             \
    {                                                                                                                  \
        cublasStatus_t status = (func);                                                                                \
        if (status != CUBLAS_STATUS_SUCCESS)                                                                           \
        {                                                                                                              \
            std::string error = "[partition_sharding_scaling] cuBLAS API failed at " + std::string(__FILE__) + ":"     \
                                + std::to_string(__LINE__) + " with status: " + std::to_string((int)status) + "\n";    \
            throw std::runtime_error(error);                                                                           \
        }                                                                                                              \
    }

// Embedding element type. Kept as float for simplicity since this project focuses on dispatch /
// partition / sharding orchestration overhead, not on numerical precision tricks (see
// embedding_quantization/ or bf16_vs_fp32/ for that).
using EMB_T = float;

class Timer
{
public:
    void tic() { start_ = std::chrono::high_resolution_clock::now(); }

    double tocMs()
    {
        auto                      stop     = std::chrono::high_resolution_clock::now();
        std::chrono::microseconds duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start_);
        return duration.count() / 1000.0;
    }

private:
    std::chrono::high_resolution_clock::time_point start_;
};

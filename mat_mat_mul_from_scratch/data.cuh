#pragma once

#include <cstdint>

namespace MatMatMulFromScratch
{

constexpr int kBlockSize = 256;

struct Data
{
    int M;
    int N;
    int K;
    float* d_A = nullptr;
    float* d_B = nullptr;
    float* h_C = nullptr;
    float* d_C = nullptr;
};

__device__ __host__ inline uint64_t getMemAddrRowMajor(int rowIdx, int colIdx, int numRows, int numCols)
{
    return (uint64_t)(rowIdx * numCols + colIdx);
}

__device__ __host__ inline uint64_t getMemAddrColMajor(int rowIdx, int colIdx, int numRows, int numCols)
{
    return (uint64_t)(colIdx * numRows + rowIdx);
}

__device__ __host__ inline uint64_t getMemAddrA(int m, int k, int M, int K)
{
    return getMemAddrRowMajor(m, k, M, K);
}

__device__ __host__ inline uint64_t getMemAddrB(int k, int n, int K, int N)
{
    return getMemAddrRowMajor(k, n, K, N);
}

__device__ __host__ inline uint64_t getMemAddrC(int m, int n, int M, int N)
{
    return getMemAddrRowMajor(m, n, M, N);
}

Data genData(int M, int N, int K);

void freeData(Data& data);

} // namespace BatchScalability
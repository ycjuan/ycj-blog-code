#pragma once

#include <cstddef>
#include <cuda_runtime.h>

enum class MemLayout
{
    ROW_MAJOR,
    COL_MAJOR
};

inline __device__ __host__ size_t getMemAddrRowMajor(size_t rowIdx, size_t colIdx, size_t numRows, size_t numCols)
{
    return rowIdx * numCols + colIdx;
}

inline __device__ __host__ size_t getMemAddrColMajor(size_t rowIdx, size_t colIdx, size_t numRows, size_t numCols)
{
    return colIdx * numRows + rowIdx;
}
#pragma once

#include <cstdint>
#include <vector>

typedef int32_t ABM_DATA_TYPE;

template <typename T>
struct CudaDeleter
{
    void operator()(T* ptr) const noexcept
    {
        cudaFree(ptr);
        ptr = nullptr;
    }
};

__device__ __host__ inline uint64_t getMemAddr(int row, int col, int numRows, int numCols)
{
    return (uint64_t)row * numCols + col;
    //return (uint64_t)col * numRows + row;
}

class AbmDataGpu
{
public:

    void init(const std::vector<std::vector<std::vector<ABM_DATA_TYPE>>> &data3D, bool useManagedMemory = false);

    void free();

    // -----------------------
    // Getters
    __device__ __host__ ABM_DATA_TYPE getVal(int row, int offset) const
    {
        return m_d_data[getMemAddrVal(row, offset)];
    }

    __device__ __host__ ABM_DATA_TYPE getOffset(int row, int field) const
    {
        return m_d_data[getMemAddrOffset(row, field)];
    }

private:
    // For example, let's say we have a 3D array like this:
    //   [ [1, 2], [3, 4, 5]
    //     [11, 12, 13], [14, 15, 16, 17] ]
    // Assuming numValPerRow is 7, it will be stored as:
    //   m_d_data: [1, 2, 3, 4, 5, 0, 0, 11, 12, 13, 14, 15, 16, 17]
    //   m_d_offsets: [0, 2, 5, 0, 3, 7]
    ABM_DATA_TYPE *m_d_data = nullptr;
    uint64_t m_d_data_size = 0;
    uint64_t m_d_data_size_in_bytes = 0;

    uint32_t m_numRows = 0;
    uint32_t m_numFields = 0;
    uint32_t m_maxNumValsPerRow = 0;

    __device__ __host__ uint64_t getMemAddrVal(int row, int offset) const
    {
        return getMemAddr(row, m_numFields + 1 + offset, m_numRows, m_maxNumValsPerRow);
    }

    __device__ __host__ uint64_t getMemAddrOffset(int row, int field) const
    {
        return getMemAddr(row, field, m_numRows, m_maxNumValsPerRow);
    }
};

std::vector<std::vector<std::vector<ABM_DATA_TYPE>>>
genRandData3D(int numRows, int numFields, std::vector<int> numValsPerFieldMin, std::vector<int> numValsPerFieldMax);
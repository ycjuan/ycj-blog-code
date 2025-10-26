#pragma once

#include <cstdint>
#include <memory>
#include <vector>

template <typename T>
struct CudaDeleter
{
    void operator()(T* ptr) const noexcept
    {
        cudaFree(ptr);
        ptr = nullptr;
    }
};

__device__ __host__ inline uint64_t getMemAddrRowMajorDevice(int row, int col, int numRows, int numCols)
{
    return (uint64_t)row * numCols + col;
}

class AbmDataGpu
{
public:

    void init(const std::vector<std::vector<std::vector<long>>> &data3D, bool useManagedMemory = false);

    void free();

    // -----------------------
    // Getters
    __device__ __host__ uint64_t getVal_d(int row, int offset) const
    {
        return m_d_data[getMemAddrData_dh(row, offset)];
    }

    __device__ __host__ uint32_t getOffset_d(int row, int field) const
    {
        return m_d_offsets[getMemAddrOffsets_dh(row, field)];
    }

private:
    // For example, let's say we have a 3D array like this:
    //   [ [1, 2], [3, 4, 5]
    //     [11, 12, 13], [14, 15, 16, 17] ]
    // Assuming numValPerRow is 7, it will be stored as:
    //   m_d_data: [1, 2, 3, 4, 5, 0, 0, 11, 12, 13, 14, 15, 16, 17]
    //   m_d_offsets: [0, 2, 5, 0, 3, 7]
    long *m_d_data = nullptr;
    uint32_t *m_d_offsets = nullptr;
    const uint64_t m_k_d_data_size = 0;
    const uint64_t m_k_d_offsets_size = 0;
    const uint64_t m_k_d_data_size_in_bytes = 0;
    const uint64_t m_k_d_offsets_size_in_bytes = 0;

    const uint32_t m_kNumRows = 0;
    const uint32_t m_kNumFields = 0;
    const uint32_t m_kMaxNumValsPerRow = 0;

    __device__ __host__ uint64_t getMemAddrData_dh(int row, int offset) const
    {
        return getMemAddrRowMajorDevice(row, offset, m_kNumRows, m_kMaxNumValsPerRow);
    }

    __device__ __host__ uint64_t getMemAddrOffsets_dh(int row, int field) const
    {
        return getMemAddrRowMajorDevice(row, field, m_kNumRows, m_kNumFields + 1);
    }
};

std::vector<std::vector<std::vector<long>>>
genRandData3D(int numRows, int numFields, std::vector<int> numValsPerFieldMin, std::vector<int> numValsPerFieldMax);
#pragma once

#include <cstdint>
#include <vector>

typedef int32_t ABM_DATA_TYPE;

__device__ __host__ inline uint64_t getMemAddr(int row, int col, int numRows, int numCols)
{
    return (uint64_t)row * numCols + col; // row-major
    //return (uint64_t)col * numRows + row; // col-major
}

class AbmDataGpu
{
public:
    void init(const std::vector<std::vector<std::vector<ABM_DATA_TYPE>>>& data3D,
              int targetField,
              bool useManagedMemory = false);

    void free();

    // -----------------------
    // Getters
    __device__ __host__ ABM_DATA_TYPE getVal(int row, int valIdx) const
    {
        return m_d_data[getMemAddrVal(row, valIdx)];
    }

    __device__ __host__ uint32_t getNumVals(int row) const
    {
        return m_d_data[getMemAddrNumVals(row)];
    }

private:
    // For example, let's say we have a 2D array like this:
    //   [ 
    //      [7, 9],       // row 0, numVals = 2
    //      [13, 12, 15]  // row 1, numVals = 3
    //   ]
    // Assuming numValPerRow is 5, it will be stored as:
    //   m_d_data: [2, 7, 9, 0, 0, 0, 3, 13, 12, 15, 0, 0]
    ABM_DATA_TYPE *m_d_data = nullptr;
    uint64_t m_d_data_size = 0;
    uint64_t m_d_data_size_in_bytes = 0;

    uint32_t m_numRows = 0;
    uint32_t m_maxNumValsPerRow = 0;

    __device__ __host__ uint64_t getMemAddrVal(int row, int valIdx) const
    {
        return getMemAddr(row, valIdx + 1, m_numRows, m_maxNumValsPerRow);
    }

    __device__ __host__ uint64_t getMemAddrNumVals(int row) const
    {
        return getMemAddr(row, 0, m_numRows, m_maxNumValsPerRow);
    }
};

std::vector<std::vector<std::vector<ABM_DATA_TYPE>>> genRandData3D(int numRows,
                                                                   int numFields,
                                                                   std::vector<int> numValsPerFieldMin,
                                                                   std::vector<int> numValsPerFieldMax,
                                                                   std::vector<int> cardinalities);
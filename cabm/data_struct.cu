#include <stdexcept>
#include <random>
#include <algorithm>

#include "data_struct.cuh"

void AbmDataGpuOneField::init(const std::vector<std::vector<std::vector<ABM_DATA_TYPE>>>& data3D,
                              int targetField,
                              bool useManagedMemory)
{
    // -----------------
    // Check empty
    {
        if (data3D.empty())
        {
            throw std::runtime_error("data3D is empty");
        }
        m_numRows = data3D.size();
    }

    // -----------------
    // Construct 2D data and infer meta data
    std::vector<std::vector<ABM_DATA_TYPE>> data2D;
    {
        m_maxNumValsPerRow = 0;
        for (const auto& inputData2D : data3D)
        {
            const auto& inputData1D = inputData2D.at(targetField);
            m_maxNumValsPerRow = std::max(m_maxNumValsPerRow, (uint32_t)inputData1D.size());
            data2D.push_back(inputData1D);
        }
        m_maxNumValsPerRow++; // The first element is reserved for storing num vals per row
                              // So we need one additional space
    }

    // -----------------
    // Malloc data
    {   
        // -----------
        // Calculate the size of the data
        m_d_data_size = m_numRows * m_maxNumValsPerRow;
        m_d_data_size_in_bytes = m_d_data_size * sizeof(ABM_DATA_TYPE);

        // -----------
        // Malloc data
        cudaError_t cudaError;
        if (useManagedMemory)
        {
            cudaError = cudaMallocManaged(&m_d_data, m_d_data_size_in_bytes);
        }
        else
        {
            cudaError = cudaMalloc(&m_d_data, m_d_data_size_in_bytes);
        }
        if (cudaError != cudaSuccess)
        {
            throw std::runtime_error("cudaMalloc failed (data): " + std::string(cudaGetErrorString(cudaError)));
        }
    }

    // -----------------
    // Init data
    {
        // -----------------
        // Malloc pinned memory
        // TODO: Release resource when there is an exception
        ABM_DATA_TYPE *hp_data;
        cudaError_t cudaError = cudaMallocHost(&hp_data, m_d_data_size_in_bytes);
        if (cudaError != cudaSuccess)
        {
            throw std::runtime_error("cudaMallocHost failed (data): " + std::string(cudaGetErrorString(cudaError)));
        }

        // -----------------
        // Fill the data in pinned memory
        for (int row = 0; row < m_numRows; row++)
        {
            hp_data[getMemAddrNumVals(row)] = data2D.at(row).size();
            for (int valIdx = 0; valIdx < data2D.at(row).size(); valIdx++)
            {
                hp_data[getMemAddrVal(row, valIdx)] = data2D.at(row).at(valIdx);
            }
        }

        // -----------------
        // Copy data to device
        cudaError = cudaMemcpy(m_d_data, hp_data, m_d_data_size_in_bytes, cudaMemcpyHostToDevice);
        if (cudaError != cudaSuccess)
        {
            throw std::runtime_error("cudaMemcpy failed (data): " + std::string(cudaGetErrorString(cudaError)));
        }

        // -----------------
        // Free pinned memory
        cudaError = cudaFreeHost(hp_data);
        if (cudaError != cudaSuccess)
        {
            throw std::runtime_error("cudaFreeHost failed (data): " + std::string(cudaGetErrorString(cudaError)));
        }
    }
}

void AbmDataGpuOneField::free()
{
    if (m_d_data != nullptr)
    {
        cudaFree(m_d_data);
        m_d_data = nullptr;
    }
    m_d_data_size = 0;
    m_d_data_size_in_bytes = 0;
    m_numRows = 0;
    m_maxNumValsPerRow = 0;
}

std::vector<std::vector<std::vector<ABM_DATA_TYPE>>>
genRandData3D(int numRows, int numFields, std::vector<int> numValsPerFieldMin, std::vector<int> numValsPerFieldMax)
{
    // -----------------
    // Check input
    if (numValsPerFieldMin.size() != numFields)
    {
        throw std::runtime_error("numValsPerFieldMin.size() != numFields");
    }
    if (numValsPerFieldMax.size() != numFields)
    {
        throw std::runtime_error("numValsPerFieldMax.size() != numFields");
    }

    // -----------------
    // Prepare random number generator
    std::default_random_engine generator;
    std::uniform_int_distribution<ABM_DATA_TYPE> valDist;

    // -----------------
    // Generate random data
    std::vector<std::vector<std::vector<ABM_DATA_TYPE>>> data3D;
    for (int row = 0; row < numRows; row++)
    {
        std::vector<std::vector<ABM_DATA_TYPE>> data2D;
        for (int field = 0; field < numFields; field++)
        {
            std::vector<ABM_DATA_TYPE> data1D;
            std::uniform_int_distribution<int> numValsDist(numValsPerFieldMin.at(field), numValsPerFieldMax.at(field));
            int numVals = numValsDist(generator);
            for (int val = 0; val < numVals; val++)
            {
                data1D.push_back(valDist(generator));
            }
            std::sort(data1D.begin(), data1D.end());
            data2D.push_back(data1D);
        }
        data3D.push_back(data2D);
    }

    return data3D;
}
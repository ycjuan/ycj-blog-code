#include <stdexcept>
#include <random>

#include "data_struct.cuh"

void AbmDataGpu::init(const std::vector<std::vector<std::vector<long>>> &data3D, bool useManagedMemory)
{
    // -----------------
    // Check empty
    {
        if (data3D.empty())
        {
            throw std::runtime_error("data3D is empty");
        }
    
    }

    // -----------------
    // Infer meta data
    {
        int numRows = data3D.size();
        int numFields = data3D.at(0).size();
        int maxNumValsPerRow = 0;

        for (const auto &data2D : data3D)
        {
            if (data2D.size() != numFields)
            {
                throw std::runtime_error("data2D has different number of fields");
            }

            int numValsPerRow = 0;
            for (const auto &data1D : data2D)
            {
                numValsPerRow += data1D.size();
            }
            maxNumValsPerRow = std::max(maxNumValsPerRow, numValsPerRow);
        }
        
        m_maxNumValsPerRow = maxNumValsPerRow;    
        m_numRows = numRows;
        m_numFields = numFields;
    }

    // -----------------
    // Malloc data
    {   
        // -----------
        // Calculate the size of the data
        m_d_data_size = m_numRows * m_maxNumValsPerRow;
        m_d_data_size_in_bytes = m_d_data_size * sizeof(long);
        m_d_offsets_size = m_numRows * (m_numFields + 1);
        m_d_offsets_size_in_bytes = m_d_offsets_size * sizeof(uint32_t);

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
        
        if (useManagedMemory)
        {
            cudaError = cudaMallocManaged(&m_d_offsets, m_d_offsets_size_in_bytes);
        }
        else
        {
            cudaError = cudaMalloc(&m_d_offsets, m_d_offsets_size_in_bytes);
        }
        if (cudaError != cudaSuccess)
        {
            throw std::runtime_error("cudaMalloc failed (offsets): " + std::string(cudaGetErrorString(cudaError)));
        }
    }

    // -----------------
    // Init data
    {
        // -----------------
        // Malloc pinned memory
        // TODO: Release resource when there is an exception
        long *hp_data;
        cudaError_t cudaError = cudaMallocHost(&hp_data, m_d_data_size_in_bytes);
        if (cudaError != cudaSuccess)
        {
            throw std::runtime_error("cudaMallocHost failed (data): " + std::string(cudaGetErrorString(cudaError)));
        }
        uint32_t *hp_offsets;
        cudaError = cudaMallocHost(&hp_offsets, m_d_offsets_size_in_bytes);
        if (cudaError != cudaSuccess)
        {
            throw std::runtime_error("cudaMallocHost failed (offsets): " + std::string(cudaGetErrorString(cudaError)));
        }

        // -----------------
        // Fill the data in pinned memory
        for (int row = 0; row < m_numRows; row++)
        {
            int offset = 0;
            hp_offsets[getMemAddrOffsets_dh(row, 0)] = offset;
            for (int field = 0; field < m_numFields; field++)
            {
                for (auto val : data3D.at(row).at(field))
                {
                    hp_data[getMemAddrData_dh(row, offset)] = val;
                    offset++;
                }
                hp_offsets[getMemAddrOffsets_dh(row, field+1)] = offset;
            }
            hp_offsets[getMemAddrOffsets_dh(row, m_numFields)] = offset;
        }

        // -----------------
        // Copy data to device
        cudaError = cudaMemcpy(m_d_data, hp_data, m_d_data_size_in_bytes, cudaMemcpyHostToDevice);
        if (cudaError != cudaSuccess)
        {
            throw std::runtime_error("cudaMemcpy failed (data): " + std::string(cudaGetErrorString(cudaError)));
        }
        cudaError = cudaMemcpy(m_d_offsets, hp_offsets, m_d_offsets_size_in_bytes, cudaMemcpyHostToDevice);
        if (cudaError != cudaSuccess)
        {
            throw std::runtime_error("cudaMemcpy failed (offsets): " + std::string(cudaGetErrorString(cudaError)));
        }

        // -----------------
        // Free pinned memory
        cudaError = cudaFreeHost(hp_data);
        if (cudaError != cudaSuccess)
        {
            throw std::runtime_error("cudaFreeHost failed (data): " + std::string(cudaGetErrorString(cudaError)));
        }
        cudaError = cudaFreeHost(hp_offsets);
        if (cudaError != cudaSuccess)
        {
            throw std::runtime_error("cudaFreeHost failed (offsets): " + std::string(cudaGetErrorString(cudaError)));
        }
    }
}

void AbmDataGpu::free()
{
    if (m_d_data != nullptr)
    {
        cudaFree(m_d_data);
        m_d_data = nullptr;
    }
    if (m_d_offsets != nullptr)
    {
        cudaFree(m_d_offsets);
        m_d_offsets = nullptr;
    }
    m_d_data_size = 0;
    m_d_data_size_in_bytes = 0;
    m_d_offsets_size = 0;
    m_d_offsets_size_in_bytes = 0;
    m_numRows = 0;
    m_numFields = 0;
    m_maxNumValsPerRow = 0;
}

std::vector<std::vector<std::vector<long>>>
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
    std::uniform_int_distribution<long> valDist;

    // -----------------
    // Generate random data
    std::vector<std::vector<std::vector<long>>> data3D;
    for (int row = 0; row < numRows; row++)
    {
        std::vector<std::vector<long>> data2D;
        for (int field = 0; field < numFields; field++)
        {
            std::vector<long> data1D;
            std::uniform_int_distribution<int> numValsDist(numValsPerFieldMin.at(field), numValsPerFieldMax.at(field));
            int numVals = numValsDist(generator);
            for (int val = 0; val < numVals; val++)
            {
                data1D.push_back(valDist(generator));
            }
            data2D.push_back(data1D);
        }
        data3D.push_back(data2D);
    }

    return data3D;
}
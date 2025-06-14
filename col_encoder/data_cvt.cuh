#ifndef DATA_CVT_CUH
#define DATA_CVT_CUH

#include <vector>
#include <stdexcept>

#include "data_struct.cuh"

EmbData convertEmbDataToGpu(const std::vector<std::vector<std::vector<float>>> &data3D)
{
    using namespace std;

    EmbData embData;

    // -----------------
    // Some meta data
    const int numRows = data3D.size();
    const int numFields = data3D.at(0).size();
    const int embDimPerField = data3D.at(0).at(0).size();
    embData.numRows = numRows;
    embData.numFields = numFields;
    embData.embDimPerField = embDimPerField;

    // -----------------
    // Malloc buffer
    EMB_T *hp_embData;
    const size_t embDataSizeInBytes = (size_t)numRows * numFields * embDimPerField * sizeof(EMB_T);
    cudaError_t cudaError = cudaMallocHost(&hp_embData, embDataSizeInBytes);
    if (cudaError != cudaSuccess) 
    {
        throw std::runtime_error("Failed to allocate pinned memory for embedding data: " + std::to_string(cudaError));
    }

    // -----------------
    // Fill the buffer with data
    for (int rowIdx = 0; rowIdx < numRows; ++rowIdx) 
    {
        for (int fieldIdx = 0; fieldIdx < numFields; ++fieldIdx) 
        {
            for (int embIdx = 0; embIdx < embDimPerField; ++embIdx) 
            {
                size_t memAddr = embData.getMemAddr(rowIdx, fieldIdx, embIdx);
                hp_embData[memAddr] = static_cast<EMB_T>(data3D[rowIdx][fieldIdx][embIdx]);
            }
        }
    }

    // -----------------
    // Malloc device memory
    cudaError = cudaMalloc(&embData.d_embData, embDataSizeInBytes);
    if (cudaError != cudaSuccess) 
    {
        throw std::runtime_error("Failed to allocate device memory for embedding data: " + std::to_string(cudaError));
    }

    // -----------------
    // Copy data to device
    cudaError = cudaMemcpy(embData.d_embData, hp_embData, embDataSizeInBytes, cudaMemcpyHostToDevice);
    if (cudaError != cudaSuccess) 
    {
        throw std::runtime_error("Failed to copy embedding data to device: " + std::to_string(cudaError));
    }

    // -----------------
    // Free pinned memory
    cudaError = cudaFreeHost(hp_embData);
    if (cudaError != cudaSuccess) 
    {
        throw std::runtime_error("Failed to free pinned memory for embedding data: " + std::to_string(cudaError));
    }

    // -----------------
    // Return the populated EmbData structure
    return embData;
}

ScoringTasksGpu convertScoringTasksToGpu(const std::vector<ScoringTask> &tasks)
{
    using namespace std;

    ScoringTasksGpu scoringTasksGpu;
    scoringTasksGpu.numTasks = tasks.size();

    // Malloc device memory for tasks
    cudaError_t cudaError = cudaMalloc(&scoringTasksGpu.d_tasks, scoringTasksGpu.numTasks * sizeof(ScoringTask));
    if (cudaError != cudaSuccess) 
    {
        throw std::runtime_error("Failed to allocate device memory for scoring tasks: " + std::to_string(cudaError));
    }

    // Copy tasks to device
    cudaError = cudaMemcpy(scoringTasksGpu.d_tasks, tasks.data(), scoringTasksGpu.numTasks * sizeof(ScoringTask), cudaMemcpyHostToDevice);
    if (cudaError != cudaSuccess) 
    {
        throw std::runtime_error("Failed to copy scoring tasks to device: " + std::to_string(cudaError));
    }

    return scoringTasksGpu;
}

std::vector<ScoringTask> convertScoringTasksBackToCpu(const ScoringTasksGpu &scoringTasksGpu)
{
    using namespace std;

    vector<ScoringTask> tasks(scoringTasksGpu.numTasks);

    cudaError_t cudaError = cudaMemcpy(tasks.data(), scoringTasksGpu.d_tasks, scoringTasksGpu.numTasks * sizeof(ScoringTask), cudaMemcpyDeviceToHost);
    if (cudaError != cudaSuccess)
    {
        throw std::runtime_error("Failed to copy scoring tasks from device to host: " + std::to_string(cudaError));
    }

    return tasks;
}

#endif // DATA_CVT_CUH
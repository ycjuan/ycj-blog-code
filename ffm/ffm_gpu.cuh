#ifndef FFM_GPU_CUH
#define FFM_GPU_CUH

#include <sstream>
#include <stdexcept>

#include "data_struct.cuh"

__global__ void ffmStep1Kernel(FFMData reqData, FFMData docData, ScoringTasksGpu tasks, float *d_buffer)
{
    int idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t taskIdx = idx / reqData.numFields;
    size_t reqFieldIdx = idx % reqData.numFields;

    if (taskIdx < tasks.numTasks && reqFieldIdx < reqData.numFields)
    {
        ScoringTask &task = tasks.d_tasks[taskIdx];

        float maxSim = -10000000.0f; // using a large negative value as a substitute for lowest float
        for (int docFieldIdx = 0; docFieldIdx < docData.numFields; ++docFieldIdx)
        {
            float sim = 0.0f;
            for (int embIdx = 0; embIdx < reqData.embDimPerField; ++embIdx)
            {
                size_t reqAddr = reqData.getMemAddr(task.reqIdx, reqFieldIdx, embIdx);
                size_t docAddr = docData.getMemAddr(task.docIdx, docFieldIdx, embIdx);

                EMB_T reqVal = reqData.d_embData[reqAddr];
                EMB_T docVal = docData.d_embData[docAddr];
                EMB_T product = reqVal * docVal;

                sim += static_cast<float>(product);
            }
            maxSim = fmaxf(maxSim, sim);
        }

        d_buffer[taskIdx * reqData.numFields + reqFieldIdx] = maxSim;
    }
}

__global__ void ffmStep2Kernel(FFMData reqData, ScoringTasksGpu tasks, float *d_buffer)
{
    int taskIdx = blockIdx.x * blockDim.x + threadIdx.x;

    if (taskIdx < tasks.numTasks)
    {
        ScoringTask &task = tasks.d_tasks[taskIdx];
        task.result = 0.0f;
        for (int reqFieldIdx = 0; reqFieldIdx < reqData.numFields; ++reqFieldIdx)
        {
            task.result += d_buffer[taskIdx * reqData.numFields + reqFieldIdx];
        }
    }

}

void ffmScorerGpu(FFMData reqData, FFMData docData, ScoringTasksGpu tasks, float *d_buffer)
{
    using namespace std;
    
    // Launch the FFM kernel - step1
    int blockSize = 256;
    int numBlocks = (tasks.numTasks * reqData.numFields + blockSize - 1) / blockSize;
    ffmStep1Kernel<<<numBlocks, blockSize>>>(reqData, docData, tasks, d_buffer);
    cudaError_t cudaError = cudaDeviceSynchronize();
    if (cudaError != cudaSuccess)
    {
        ostringstream oss;
        oss << "CUDA error in ffmScorerGpu (step 1): " << cudaGetErrorString(cudaError);
        throw runtime_error(oss.str());
    }

    // Launch the FFM kernel - step2
    numBlocks = (tasks.numTasks + blockSize - 1) / blockSize;
    ffmStep2Kernel<<<numBlocks, blockSize>>>(reqData, tasks, d_buffer);
    cudaError = cudaDeviceSynchronize();
    if (cudaError != cudaSuccess)
    {
        ostringstream oss;
        oss << "CUDA error in ffmScorerGpu (step 2): " << cudaGetErrorString(cudaError);
        throw runtime_error(oss.str());
    }
}

#endif // FFM_GPU_CUH
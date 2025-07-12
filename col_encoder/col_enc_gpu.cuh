#ifndef ColEnc_GPU_CUH
#define ColEnc_GPU_CUH

#include <sstream>
#include <stdexcept>

#include "data_struct.cuh"

__global__ void colEncStep1Kernel(ColEncData reqData, ColEncData docData, ScoringTasksGpu tasks, float *d_buffer)
{
    int idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t taskIdx = idx / reqData.numFields;
    size_t reqFieldIdx = idx % reqData.numFields;

    if (taskIdx < tasks.numTasks && reqFieldIdx < reqData.numFields)
    {
        ScoringTask &task = tasks.d_tasks[taskIdx];

        float sim = 0.0f;
        for (int docFieldIdx = 0; docFieldIdx < docData.numFields; ++docFieldIdx)
        {
            for (int embIdx = 0; embIdx < reqData.embDimPerField; ++embIdx)
            {
                size_t reqAddr = reqData.getMemAddr(task.reqIdx, reqFieldIdx, embIdx);
                size_t docAddr = docData.getMemAddr(task.docIdx, docFieldIdx, embIdx);

                EMB_T reqVal = reqData.d_embData[reqAddr];
                EMB_T docVal = docData.d_embData[docAddr];
                EMB_T product = reqVal * docVal;

                sim += static_cast<float>(product);
            }
        }

        d_buffer[taskIdx * reqData.numFields + reqFieldIdx] = sim;
    }
}

__global__ void colEncStep2Kernel(ColEncData reqData, ScoringTasksGpu tasks, float *d_buffer)
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

void colEncScorerGpu(ColEncData reqData, ColEncData docData, ScoringTasksGpu tasks, float *d_buffer)
{
    using namespace std;
    
    // Launch the ColEnc kernel - step1
    int blockSize = 256;
    int numBlocks = (tasks.numTasks * reqData.numFields + blockSize - 1) / blockSize;
    colEncStep1Kernel<<<numBlocks, blockSize>>>(reqData, docData, tasks, d_buffer);
    cudaError_t cudaError = cudaDeviceSynchronize();
    if (cudaError != cudaSuccess)
    {
        ostringstream oss;
        oss << "CUDA error in colEncScorerGpu (step 1): " << cudaGetErrorString(cudaError);
        throw runtime_error(oss.str());
    }

    // Launch the ColEnc kernel - step2
    numBlocks = (tasks.numTasks + blockSize - 1) / blockSize;
    colEncStep2Kernel<<<numBlocks, blockSize>>>(reqData, tasks, d_buffer);
    cudaError = cudaDeviceSynchronize();
    if (cudaError != cudaSuccess)
    {
        ostringstream oss;
        oss << "CUDA error in colEncScorerGpu (step 2): " << cudaGetErrorString(cudaError);
        throw runtime_error(oss.str());
    }
}

#endif // ColEnc_GPU_CUH
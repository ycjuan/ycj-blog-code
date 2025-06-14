#ifndef COL_ENCODER_GPU_CUH
#define COL_ENCODER_GPU_CUH

#include <sstream>
#include <stdexcept>

#include "data_struct.cuh"

__global__ void colEncoderKernel(EmbData reqData, EmbData docData, ScoringTasksGpu tasks)
{
    int taskIdx = blockIdx.x * blockDim.x + threadIdx.x;

    if (taskIdx < tasks.numTasks)
    {
        ScoringTask &task = tasks.d_tasks[taskIdx];
        task.result = 0.0f;

        for (int reqFieldIdx = 0; reqFieldIdx < reqData.numFields; ++reqFieldIdx) // swapping req / doc loops may be 10% faster
        {
            for (int docFieldIdx = 0; docFieldIdx < docData.numFields; ++docFieldIdx)
            {
                for (int embIdx = 0; embIdx < reqData.embDimPerField; ++embIdx)
                {
                    size_t reqAddr = reqData.getMemAddr(task.reqIdx, reqFieldIdx, embIdx);
                    size_t docAddr = docData.getMemAddr(task.docIdx, docFieldIdx, embIdx);

                    EMB_T reqVal = reqData.d_embData[reqAddr];
                    EMB_T docVal = docData.d_embData[docAddr];
                    EMB_T product = reqVal * docVal;

                    task.result += static_cast<float>(product);
                }
            }
        }
    }
}

void colEncoderScorerGpu(EmbData reqData, EmbData docData, ScoringTasksGpu tasks)
{
    using namespace std;
    
    int blockSize = 256;
    int numBlocks = (tasks.numTasks + blockSize - 1) / blockSize;
    colEncoderKernel<<<numBlocks, blockSize>>>(reqData, docData, tasks);
    cudaError_t cudaError = cudaDeviceSynchronize();
    if (cudaError != cudaSuccess)
    {
        ostringstream oss;
        oss << "CUDA error in colEncoderScorerGpu: " << cudaGetErrorString(cudaError);
        throw runtime_error(oss.str());
    }
}

#endif // COL_ENCODER_GPU_CUH
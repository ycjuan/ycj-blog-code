#ifndef COL_ENCODER_GPU_CUH
#define COL_ENCODER_GPU_CUH

#include <sstream>
#include <stdexcept>
#include <limits>

#include "data_struct.cuh"
#include "util.cuh"

__global__ void h2dKernel(EmbData docData, ScoringTasksGpu tasks, int numToCopy)
{
    int taskIdx = blockIdx.x * blockDim.x + threadIdx.x;

    if (taskIdx < numToCopy)
    {
        ScoringTask &task = tasks.d_tasks[taskIdx];
        for (int fieldIdx = 0; fieldIdx < docData.numFields; ++fieldIdx)
        {
            for (int embIdx = 0; embIdx < docData.embDimPerField; ++embIdx)
            {
                size_t addr = docData.getMemAddr(task.docIdx, fieldIdx, embIdx);
                EMB_T value = docData.hp_embData[addr];
                docData.d_embData[addr] = value;
            }
        }
    }
}

__global__ void colEncoderKernel(EmbData reqData, EmbData docData, ScoringTasksGpu tasks)
{
    int taskIdx = blockIdx.x * blockDim.x + threadIdx.x;

    if (taskIdx < tasks.numTasks)
    {
        ScoringTask &task = tasks.d_tasks[taskIdx];
        task.result = 0.0f;

        for (int reqFieldIdx = 0; reqFieldIdx < reqData.numFields; ++reqFieldIdx) // swapping req / doc loops may be 10% faster
        {
            // float maxSim = std::numeric_limits<float>::lowest(); // can't compile, will figure out later
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
            task.result += maxSim;
        }
    }
}

struct ColEncoderScorerRst
{
    float copyTimeMs = 0.0f;
    float scoringTimeMs = 0.0f;
    float totalTimeMs = 0.0f;
};

ColEncoderScorerRst colEncoderScorerGpu(EmbData reqData, EmbData docData, ScoringTasksGpu tasks, float h2dRatio)
{
    using namespace std;

    ColEncoderScorerRst rst;
    const int numToCopy = tasks.numTasks * h2dRatio;
    const int kBlockSize = 256;

    // ---------------
    // H2D copy
    if (numToCopy > 0)
    {
        CudaTimer h2dTimer;
        h2dTimer.tic();
        int numBlocks = (numToCopy + kBlockSize - 1) / kBlockSize;
        h2dKernel<<<numBlocks, kBlockSize>>>(docData, tasks, numToCopy);
        cudaError_t cudaError = cudaDeviceSynchronize();
        if (cudaError != cudaSuccess)
        {
            ostringstream oss;
            oss << "CUDA error in h2dKernel: " << cudaGetErrorString(cudaError);
            throw runtime_error(oss.str());
        }
        rst.copyTimeMs = h2dTimer.tocMs();
    }

    // ---------------
    // Scoring
    {
        int numBlocks = (tasks.numTasks + kBlockSize - 1) / kBlockSize;
        colEncoderKernel<<<numBlocks, kBlockSize>>>(reqData, docData, tasks);
        cudaError_t cudaError = cudaDeviceSynchronize();
        if (cudaError != cudaSuccess)
        {
            ostringstream oss;
            oss << "CUDA error in colEncoderKernel: " << cudaGetErrorString(cudaError);
            throw runtime_error(oss.str());
        }
    }

    return rst;
}

#endif // COL_ENCODER_GPU_CUH
#ifndef COLENC_GPU_CUH
#define COLENC_GPU_CUH

#include <sstream>
#include <stdexcept>
#include <sstream>
#include <string>
#include <cublas_v2.h>

#include "data_struct.cuh"

using namespace std;

enum class MemLayout
{
    ROW_MAJOR,
    COL_MAJOR
};

constexpr MemLayout reqMemLayout = MemLayout::ROW_MAJOR;
constexpr MemLayout docMemLayout = MemLayout::ROW_MAJOR;

struct CublasGemmExParam
{
    int numTasks; // This is numToScore * numFields
    int numFields;
    int embDim;
    void *d_doc = nullptr; // M=numTasks x N=embDim
    void *d_req = nullptr; // M=numFields x N=embDim
    void *d_rst = nullptr; // M=numTasks x N=numFields
    cublasComputeType_t computeType = CUBLAS_COMPUTE_32F;
    cudaDataType dataType = CUDA_R_16BF;
    const cublasHandle_t *p_cublasHandle;
};

void cublasGemmExWrapper(CublasGemmExParam &data)
{
    const float alpha = 1.0;
    const float beta = 0.0;

    int M = data.numTasks;
    int N = data.numFields;
    int K = data.embDim;

    void *matA = data.d_doc;
    void *matB = data.d_req;
    void *matC = data.d_rst;

    cublasOperation_t tranA = (docMemLayout == MemLayout::COL_MAJOR) ? CUBLAS_OP_N : CUBLAS_OP_T;
    cublasOperation_t tranB = (reqMemLayout == MemLayout::COL_MAJOR) ? CUBLAS_OP_T : CUBLAS_OP_N;

    int ldA = (docMemLayout == MemLayout::COL_MAJOR) ? M : K;
    int ldB = (reqMemLayout == MemLayout::COL_MAJOR) ? N : K;

    const cublasGemmAlgo_t kCublasGemmAlgo = CUBLAS_GEMM_DEFAULT; // According to https://docs.nvidia.com/cuda/cublas/#cublasmath-t, this is deprecated after A100
    const cublasDataType_t kDataTypeC = CUDA_R_32F;               // We always use FP32 for matrix C
    cublasStatus_t status = cublasGemmEx(*data.p_cublasHandle, tranA, tranB,
                                         M, N, K,
                                         &alpha,
                                         matA, data.dataType, ldA,
                                         matB, data.dataType, ldB,
                                         &beta,
                                         matC, kDataTypeC, M,
                                         data.computeType, kCublasGemmAlgo);

    if (status != CUBLAS_STATUS_SUCCESS)
    {
        ostringstream oss;
        oss << "cublasGemmEx failed with error: " << to_string(status);
        throw std::runtime_error(oss.str());
    }
}

__global__ void densifyKernel(ColEncData docData, ScoringTasksGpu tasks, size_t reqIdx, size_t numToScore, EMB_T *d_tmpDocData)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t quotient1 = idx / (docData.numFields * docData.embDimPerField);
    size_t remainder1 = idx % (docData.numFields * docData.embDimPerField);
    size_t quotient2 = remainder1 / docData.embDimPerField;
    size_t remainder2 = remainder1 % docData.embDimPerField;

    size_t densifiedDocIdx = quotient1;
    size_t fieldIdx = quotient2;
    size_t embIdx = remainder2;
    size_t taskIdx = reqIdx * numToScore + densifiedDocIdx;

    if (densifiedDocIdx < numToScore && fieldIdx < docData.numFields && taskIdx < tasks.numTasks)
    {
        ScoringTask &task = tasks.d_tasks[taskIdx];
        size_t docIdx = task.docIdx;

        size_t srcMemAddr = docData.getMemAddr(docIdx, fieldIdx, embIdx);
        size_t dstMemAddr = densifiedDocIdx * docData.numFields * docData.embDimPerField + fieldIdx * docData.embDimPerField + embIdx;
        d_tmpDocData[dstMemAddr] = docData.d_embData[srcMemAddr];
    }
}

__global__ void mergeKernel(ColEncData docData, ScoringTasksGpu tasks, size_t reqIdx, size_t numToScore, float *d_tmpRst)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t densifiedDocIdx = idx;
    size_t taskIdx = reqIdx * numToScore + densifiedDocIdx;
    size_t numFields = docData.numFields;

    if (taskIdx < tasks.numTasks && densifiedDocIdx < numToScore)
    {
        ScoringTask &task = tasks.d_tasks[taskIdx];
        task.result = 0.0f;

        for (int reqFieldIdx = 0; reqFieldIdx < numFields; ++reqFieldIdx)
        {
            // float maxSim = std::numeric_limits<float>::lowest(); // can't compile, will figure out later
            float maxSim = -10000000.0f; // using a large negative value as a substitute for lowest float
            for (int docFieldIdx = 0; docFieldIdx < numFields; ++docFieldIdx)
            {
                 // cublasGemmEx output is always col-major
                size_t memAddr = reqFieldIdx * (numToScore * numFields) + densifiedDocIdx * numFields + docFieldIdx;
                float sim = d_tmpRst[memAddr];
                maxSim = fmaxf(maxSim, sim);
            }
            task.result += maxSim;
        }
    }
}

void colEncScorerGpu(ColEncData reqData,
                     ColEncData docData,
                     ScoringTasksGpu tasks,
                     cudaStream_t stream,
                     const cublasHandle_t &cublasHandle,
                     EMB_T *d_tmpDocData,
                     float *d_tmpRst)
{
    using namespace std;

    const size_t kNumReqs = reqData.numRows;
    const size_t kNumToScorePerReq = tasks.numTasks / kNumReqs;
    const size_t kNumFields = reqData.numFields;
    const size_t kEmbDim = reqData.embDimPerField;

    // ------------------
    // Perform scoring
    {
        for (int reqIdx = 0; reqIdx < kNumReqs; ++reqIdx)
        {
            // ---------------
            // Step 1 - copy document data to temporary storage
            {
                // Copy the document data for the current request to temporary storage
                int blockSize = 256;
                size_t numBlocks = ((size_t)kNumToScorePerReq * kNumFields * kEmbDim + blockSize - 1) / blockSize;
                densifyKernel<<<numBlocks, blockSize, 0, stream>>>(docData, tasks, reqIdx, kNumToScorePerReq, d_tmpDocData);
                cudaError_t cudaError = cudaStreamSynchronize(stream);
                if (cudaError != cudaSuccess)
                {
                    ostringstream oss;
                    oss << "CUDA error in colEncoderScorerGpu (copying document data): " << cudaGetErrorString(cudaError);
                    throw runtime_error(oss.str());
                }
            }

            // ---------------
            // Step 2 - perform scoring with cublasGemmEx
            {
                CublasGemmExParam cublasParam;
                cublasParam.numTasks = kNumToScorePerReq * kNumFields;
                cublasParam.numFields = kNumFields;
                cublasParam.embDim = kEmbDim;
                cublasParam.d_doc = d_tmpDocData;
                cublasParam.d_req = reqData.d_embData + reqIdx * kNumFields * kEmbDim;
                cublasParam.d_rst = d_tmpRst;
                cublasParam.p_cublasHandle = &cublasHandle;

                cublasSetStream(cublasHandle, stream);
                cublasGemmExWrapper(cublasParam);
            }

            // ---------------
            // Step 3 - merging results
            {
                int blockSize = 256;
                int numBlocks = (kNumToScorePerReq + blockSize - 1) / blockSize;
                mergeKernel<<<numBlocks, blockSize, 0, stream>>>(docData, tasks, reqIdx, kNumToScorePerReq, d_tmpRst);
                cudaError_t cudaError = cudaStreamSynchronize(stream);
                if (cudaError != cudaSuccess)
                {
                    ostringstream oss;
                    oss << "CUDA error in colEncoderScorerGpu (merging results): " << cudaGetErrorString(cudaError);
                    throw runtime_error(oss.str());
                }
            }
        }
    }
}

#endif
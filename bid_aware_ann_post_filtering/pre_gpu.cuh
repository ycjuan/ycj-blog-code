#ifndef PRE_GPU_CUH
#define PRE_GPU_CUH

#include "data_struct.cuh"
#include "common.cuh"
#include "topk.cuh"
#include "misc.cuh"

#include <random>
#include <iostream>
#include <algorithm>
#include <thrust/copy.h>
#include <thrust/execution_policy.h>

using namespace std;

__global__ void filterKernel(GpuAlgoParam param)
{
    size_t workerIdx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;

    if (workerIdx < param.rqData.numActivePairs)
    {
        ReqDocPair pair = param.rqData.d_data[workerIdx];
        ItemGpu req = param.reqData.d_item[pair.reqIdx];
        ItemGpu doc = param.docData.d_item[pair.docIdx];
        pair.score = (doc.randAttr <= req.randAttr) ? 1.0f : 0.0f;
        param.reqBufferData.d_data[workerIdx] = pair;
    }
}

__global__ void scoringKernel(GpuAlgoParam param)
{
    size_t workerIdx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;

    if (workerIdx < param.rqData.numActivePairs)
    {
        ReqDocPair &pair = param.rqData.d_data[workerIdx];
        pair.score = getScoreDevice(param.reqData, param.docData, pair.reqIdx, pair.docIdx);
        pair.score *= param.docData.d_item[pair.docIdx].bid;
    }
}

struct NonZeroPredicator
{
    __host__ __device__ bool operator()(const ReqDocPair x)
    {
        return x.score != 0;
    }
};

vector<ReqDocPair> preGpuAlgoSingle(GpuAlgoParam param)
{
    int blockSize = 256;
    int gridSize = (param.rqData.numActivePairs + blockSize - 1) / blockSize;

    filterKernel<<<gridSize, blockSize>>>(param);
    cudaDeviceSynchronize();
    cudaError_t cudaError = cudaGetLastError();
    if (cudaError != cudaSuccess)
    {
        throw runtime_error("filterKernel failed: " + string(cudaGetErrorString(cudaError)));
    }
    param.reqBufferData.numActivePairs = param.rqData.numActivePairs;

    ReqDocPair *d_endPtr = thrust::copy_if(thrust::device,
                                           param.reqBufferData.d_data,
                                           param.reqBufferData.d_data + param.reqBufferData.numActivePairs,
                                           param.rqData.d_data,
                                           NonZeroPredicator());

    param.rqData.numActivePairs = d_endPtr - param.rqData.d_data;

    gridSize = (param.rqData.numActivePairs + blockSize - 1) / blockSize;
    scoringKernel<<<gridSize, blockSize>>>(param);
    cudaDeviceSynchronize();
    cudaError = cudaGetLastError();
    if (cudaError != cudaSuccess)
    {
        throw runtime_error("scoringKernel failed: " + string(cudaGetErrorString(cudaError)));
    }

    vector<ReqDocPair> rst = param.topk.retrieveTopk(param.rqData.d_data,
                                                     param.reqBufferData.d_data,
                                                     param.rqData.numActivePairs,
                                                     param.numToRetrieve,
                                                     param.topkTimeMs);
    return rst;
}

vector<vector<ReqDocPair>> preGpuAlgoBatch(const vector<ItemCpu> &reqs, const vector<ItemCpu> &docs, int k)
{
    // prepare topk
    float maxScore;
    float minScore;
    getUpperAndLowerBound(reqs, docs, minScore, maxScore);
    assert(minScore < maxScore);

    // prepare GpuAlgoParam
    GpuAlgoParam param;
    param.docData.init(docs);
    param.numToRetrieve = k;
    param.topk.init(minScore, maxScore);
    param.reqData.init(reqs);
    
    vector<vector<ReqDocPair>> rst2D;
    for (int reqIdx = 0; reqIdx < reqs.size(); reqIdx++)
    {
        param.rqData.init(reqs[reqIdx], docs);
        param.reqBufferData.init(reqs[reqIdx], docs);

        rst2D.push_back(preGpuAlgoSingle(param));

        param.rqData.reset();
        param.reqBufferData.reset();
    }
    
    param.reqData.reset();
    param.docData.reset();

    return rst2D;
}

#endif // PRE_GPU_CUH
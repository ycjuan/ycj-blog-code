#include "data_struct.cuh"
#include "common.cuh"
#include "topk.cuh"
#include "misc.cuh"
#include "util.cuh"
#include "pre_gpu.cuh"

#include <random>
#include <iostream>
#include <algorithm>
#include <thrust/copy.h>
#include <thrust/execution_policy.h>

using namespace std;

struct PreGpuAlgoParam
{
    ItemDataGpu reqData;
    ItemDataGpu docData;
    ReqDocPairDataGpu rstData;
    ReqDocPairDataGpu bufData;
    Topk topk;
    int numToRetrieve;
    int reqIdx;
    int activePairSize;
    float timeMsFilter;
    float timeMsCopyIf;
    float timeMsScoring;
    float timeMsTopkUpdateCounter = 0;
    float timeMsTopkCopyCounterToCpu = 0;
    float timeMsTopkFindLowestBucket = 0;
    float timeMsTopkPrefilter = 0;
    float timeMsTopkSort = 0;
    float timeMsTopkCopyBackToCpu = 0;
    float timeMsTopkTotal;
    float timeMsTotal;
};

__global__ void filterKernel(PreGpuAlgoParam param)
{
    size_t docIdx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;

    if (docIdx < param.docData.numItems)
    {
        ItemGpu req = param.reqData.d_item[param.reqIdx];
        ItemGpu doc = param.docData.d_item[docIdx];
        
        ReqDocPair pair;
        pair.reqIdx = param.reqIdx;
        pair.docIdx = docIdx;
        pair.reqCentroidId = req.centroidId;
        pair.docCentroidId = doc.centroidId;
        pair.score = (doc.randAttr <= req.randAttr) ? 1.0f : 0.0f;

        param.bufData.d_data[docIdx] = pair;
    }
}

__global__ void scoringKernel(PreGpuAlgoParam param)
{
    size_t pairIdx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;

    if (pairIdx < param.activePairSize)
    {
        ReqDocPair &pair = param.rstData.d_data[pairIdx];
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

vector<ReqDocPair> preGpuAlgoSingle(PreGpuAlgoParam &param)
{
    CudaTimer timerGlobal;
    CudaTimer timerLocal;
    timerGlobal.tic();

    param.activePairSize = param.docData.numItems;
    int blockSize = 256;
    int gridSize = (param.activePairSize + blockSize - 1) / blockSize;

    // step1 - filtering
    timerLocal.tic();
    filterKernel<<<gridSize, blockSize>>>(param);
    cudaError_t cudaError = cudaDeviceSynchronize();
    if (cudaError != cudaSuccess)
    {
        throw runtime_error("filterKernel failed: " + string(cudaGetErrorString(cudaError)));
    }
    cudaError = cudaGetLastError();
    if (cudaError != cudaSuccess)
    {
        throw runtime_error("filterKernel failed (2): " + string(cudaGetErrorString(cudaError)));
    }
    param.timeMsFilter = timerLocal.tocMs();

    // step2 - copy eligible pairs
    timerLocal.tic();
    ReqDocPair *d_endPtr = thrust::copy_if(thrust::device,
                                           param.bufData.d_data,
                                           param.bufData.d_data + param.activePairSize,
                                           param.rstData.d_data,
                                           NonZeroPredicator());
    param.activePairSize = d_endPtr - param.rstData.d_data;
    param.timeMsCopyIf = timerLocal.tocMs();

    // step3 - scoring
    timerLocal.tic();
    gridSize = (param.activePairSize + blockSize - 1) / blockSize;
    scoringKernel<<<gridSize, blockSize>>>(param);
    cudaError = cudaDeviceSynchronize();
    if (cudaError != cudaSuccess)
    {
        throw runtime_error("scoringKernel failed: " + string(cudaGetErrorString(cudaError)));
    }
    cudaError = cudaGetLastError();
    if (cudaError != cudaSuccess)
    {
        throw runtime_error("scoringKernel failed (2): " + string(cudaGetErrorString(cudaError)));
    }
    param.timeMsScoring = timerLocal.tocMs();

    // step4 - topk
    TopkRetrievalParam topkParam;
    topkParam.d_pair = param.rstData.d_data;
    topkParam.d_buffer = param.bufData.d_data;
    topkParam.numReqDocPairs = param.activePairSize;
    topkParam.numToRetrieve = param.numToRetrieve;
    vector<ReqDocPair> rst = param.topk.retrieveTopk(topkParam);

    param.timeMsTopkUpdateCounter = topkParam.timeMsUpdateCounter;
    param.timeMsTopkCopyCounterToCpu = topkParam.timeMsCopyCounterToCpu;
    param.timeMsTopkFindLowestBucket = topkParam.timeMsFindLowestBucket;
    param.timeMsTopkPrefilter = topkParam.timeMsPrefilter;
    param.timeMsTopkSort = topkParam.timeMsSort;
    param.timeMsTopkCopyBackToCpu = topkParam.timeMsCopyBackToCpu;
    param.timeMsTopkTotal = topkParam.timeMsTotal;

    param.timeMsTotal = timerGlobal.tocMs();
    
    return rst;
}

vector<vector<ReqDocPair>> preGpuAlgoBatch(const vector<ItemCpu> &reqs, const vector<ItemCpu> &docs, int numToRetrieve)
{
    // prepare topk
    float maxScore;
    float minScore;
    getUpperAndLowerBound(reqs, docs, minScore, maxScore);
    assert(minScore < maxScore);

    // prepare PreGpuAlgoParam
    PreGpuAlgoParam param;
    param.docData.init(docs);
    param.numToRetrieve = numToRetrieve;
    param.topk.init(minScore, maxScore);
    param.reqData.init(reqs);
    
    vector<vector<ReqDocPair>> rst2D;

    vector<float> timeMsFilter1D;
    vector<float> timeMsCopyIf1D;
    vector<float> timeMsScoring1D;
    vector<float> timeMsTopkUpdateCounter1D;
    vector<float> timeMsTopkCopyCounterToCpu1D;
    vector<float> timeMsTopkFindLowestBucket1D;
    vector<float> timeMsTopkPrefilter1D;
    vector<float> timeMsTopkSort1D;
    vector<float> timeMsTopkCopyBackToCpu1D;
    vector<float> timeMsTopkTotal1D;
    vector<float> timeMsTotal1D;

    param.rstData.malloc(docs.size());
    param.bufData.malloc(docs.size());
    for (int reqIdx = 0; reqIdx < reqs.size(); reqIdx++)
    {
        param.reqIdx = reqIdx;
        rst2D.push_back(preGpuAlgoSingle(param));

        timeMsFilter1D.push_back(param.timeMsFilter);
        timeMsCopyIf1D.push_back(param.timeMsCopyIf);
        timeMsScoring1D.push_back(param.timeMsScoring);
        timeMsTopkUpdateCounter1D.push_back(param.timeMsTopkUpdateCounter);
        timeMsTopkCopyCounterToCpu1D.push_back(param.timeMsTopkCopyCounterToCpu);
        timeMsTopkFindLowestBucket1D.push_back(param.timeMsTopkFindLowestBucket);
        timeMsTopkPrefilter1D.push_back(param.timeMsTopkPrefilter);
        timeMsTopkSort1D.push_back(param.timeMsTopkSort);
        timeMsTopkCopyBackToCpu1D.push_back(param.timeMsTopkCopyBackToCpu);
        timeMsTopkTotal1D.push_back(param.timeMsTopkTotal);
        timeMsTotal1D.push_back(param.timeMsTotal);

    }
    
    param.reqData.reset();
    param.docData.reset();

    printLatency(timeMsFilter1D, "filter");
    printLatency(timeMsCopyIf1D, "copyIf");
    printLatency(timeMsScoring1D, "scoring");
    printLatency(timeMsTopkUpdateCounter1D, "topkUpdateCounter");
    printLatency(timeMsTopkCopyCounterToCpu1D, "topkCopyCounterToCpu");
    printLatency(timeMsTopkFindLowestBucket1D, "topkFindLowestBucket");
    printLatency(timeMsTopkPrefilter1D, "topkPrefilter");
    printLatency(timeMsTopkSort1D, "topkSort");
    printLatency(timeMsTopkCopyBackToCpu1D, "topkCopyBackToCpu");
    printLatency(timeMsTopkTotal1D, "topkTotal");
    printLatency(timeMsTotal1D, "total");

    return rst2D;
}

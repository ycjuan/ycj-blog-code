#include "data_struct.cuh"
#include "common.cuh"
#include "topk.cuh"
#include "misc.cuh"
#include "util.cuh"
#include "config.cuh"
#include "post_gpu.cuh"

#include <random>
#include <iostream>
#include <algorithm>
#include <thrust/copy.h>
#include <thrust/execution_policy.h>

using namespace std;

namespace 
{
    struct PostGpuAlgoParam
    {
        ItemDataGpu reqData;
        ItemDataGpu docData;
        CentroidDataGpu centroidData;
        ReqDocPairDataGpu rstData;
        ReqDocPairDataGpu bufData;
        Topk topk;
        int numToRetrieve;
        int reqIdx;
        int activePairSize;
        float *d_centroidScore = nullptr;
        bool enableBidAware;

        float timeMsStep1ScoreCentroid;
        float timeMsStep2PreScore;
        float timeMsStep3s1TopkApproxUpdateCounter = 0;
        float timeMsStep3s2TopkApproxCopyCounterToCpu = 0;
        float timeMsStep3s3TopkApproxFindLowestBucket = 0;
        float timeMsStep3s4TopkApproxPrefilter = 0;
        float timeMsStep3s5TopkApproxCopyback = 0;    
        float timeMsStep3TopkApproxTotal;
        float timeMsStep4Filter;
        float timeMsStep5CopyIf;
        float timeMsStep6Scoring;
        float timeMsStep7s1TopkExactUpdateCounter = 0;
        float timeMsStep7s2TopkExactCopyCounterToCpu = 0;
        float timeMsStep7s3TopkExactFindLowestBucket = 0;
        float timeMsStep7s4TopkExactPrefilter = 0;
        float timeMsStep7s5TopkExactSort = 0;
        float timeMsStep7s6TopkExactCopyBackToCpu = 0;
        float timeMsStep7TopkExactTotal;
        float timeMsTotal;
    };

    __global__ void scoreCentroidKernel(PostGpuAlgoParam param)
    {
        size_t centroidIdx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;

        if (centroidIdx < param.centroidData.numCentroids)
        {
            param.d_centroidScore[centroidIdx] = getScoreDevice(param.reqData,
                                                                param.centroidData,
                                                                param.reqIdx,
                                                                centroidIdx);

            clock_t start_clock = clock();
            clock_t clock_offset = 0;
            while (clock_offset < kScoreSlowdownCycle)
            {
                clock_offset = clock() - start_clock;
            }
        }
    }

    __global__ void preScoreKernel(PostGpuAlgoParam param)
    {
        size_t docIdx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;

        if (docIdx < param.activePairSize)
        {
            ItemGpu req = param.reqData.d_item[param.reqIdx];
            ItemGpu doc = param.docData.d_item[docIdx];

            ReqDocPair pair;
            pair.reqIdx = param.reqIdx;
            pair.docIdx = docIdx;
            pair.reqCentroidId = req.centroidId;
            pair.docCentroidId = doc.centroidId;
            pair.score = param.d_centroidScore[doc.centroidId];
            if (param.enableBidAware)
                pair.score *= param.docData.d_item[docIdx].bid;

            param.rstData.d_data[docIdx] = pair;
        }
    }

    __global__ void filterKernel(PostGpuAlgoParam param)
    {
        size_t pairIdx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;

        if (pairIdx < param.docData.numItems)
        {
            ReqDocPair &pair = param.bufData.d_data[pairIdx];
            ItemGpu req = param.reqData.d_item[pair.reqIdx];
            ItemGpu doc = param.docData.d_item[pair.docIdx];
            pair.score = (doc.randAttr <= req.randAttr) ? 1.0f : 0.0f;

            clock_t start_clock = clock();
            clock_t clock_offset = 0;
            while (clock_offset < kFilterSlowdownCycle)
            {
                clock_offset = clock() - start_clock;
            }
        }
    }

    __global__ void scoringKernel(PostGpuAlgoParam param)
    {
        size_t pairIdx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;

        if (pairIdx < param.activePairSize)
        {
            ReqDocPair &pair = param.rstData.d_data[pairIdx];
            pair.score = getScoreDevice(param.reqData, param.docData, pair.reqIdx, pair.docIdx);
            pair.score *= param.docData.d_item[pair.docIdx].bid;

            clock_t start_clock = clock();
            clock_t clock_offset = 0;
            while (clock_offset < kScoreSlowdownCycle)
            {
                clock_offset = clock() - start_clock;
            }
        }
    }

    struct NonZeroPredicator
    {
        __host__ __device__ bool operator()(const ReqDocPair x)
        {
            return x.score != 0;
        }
    };

    vector<ReqDocPair> postGpuAlgoSingle(PostGpuAlgoParam &param)
    {
        CudaTimer timerGlobal;
        CudaTimer timerLocal;
        timerGlobal.tic();

        param.activePairSize = param.docData.numItems;
        int blockSize = 256;

        // Step 1 - score centroid
        timerLocal.tic();
        int gridSize = (param.centroidData.numCentroids + blockSize - 1) / blockSize;
        scoreCentroidKernel<<<gridSize, blockSize>>>(param);
        cudaDeviceSynchronize();
        cudaError_t cudaError = cudaGetLastError();
        if (cudaError != cudaSuccess)
        {
            throw runtime_error("scoringCentroidKernel failed: " + string(cudaGetErrorString(cudaError)));
        }
        param.timeMsStep1ScoreCentroid = timerLocal.tocMs();

        // Step 2 - pre-score
        timerLocal.tic();
        gridSize = (param.activePairSize + blockSize - 1) / blockSize;
        preScoreKernel<<<gridSize, blockSize>>>(param);
        cudaDeviceSynchronize();
        cudaError = cudaGetLastError();
        if (cudaError != cudaSuccess)
        {
            throw runtime_error("preScoreKernel failed: " + string(cudaGetErrorString(cudaError)));
        }
        param.timeMsStep2PreScore = timerLocal.tocMs();

        // Step 3 - pre-filter
        timerLocal.tic();
        TopkRetrievalParam topkParam;
        topkParam.d_pair = param.rstData.d_data;
        topkParam.d_buffer = param.bufData.d_data;
        topkParam.numReqDocPairs = param.activePairSize;
        topkParam.numToRetrieve = param.activePairSize * kAnnNumtoRetrieveRatio;
        param.topk.retrieveTopkApprox(topkParam); // the result is stored in d_buffer
        if (topkParam.numRetrieved > topkParam.numToRetrieve * 1.2)
        {
            double ratio = (double)topkParam.numRetrieved / topkParam.numToRetrieve;
            cout << "[Warning] numRetrieved / numToRetrieve = " << ratio << endl;
        }
        param.activePairSize = topkParam.numRetrieved;
        param.timeMsStep3s5TopkApproxCopyback = 0;
        param.timeMsStep3s1TopkApproxUpdateCounter = topkParam.timeMsUpdateCounter;
        param.timeMsStep3s2TopkApproxCopyCounterToCpu = topkParam.timeMsCopyCounterToCpu;
        param.timeMsStep3s3TopkApproxFindLowestBucket = topkParam.timeMsFindLowestBucket;
        param.timeMsStep3s4TopkApproxPrefilter = topkParam.timeMsPrefilter;
        param.timeMsStep3TopkApproxTotal = timerLocal.tocMs();

        // Step4 - filter
        timerLocal.tic();
        gridSize = (param.activePairSize + blockSize - 1) / blockSize;
        filterKernel<<<gridSize, blockSize>>>(param);
        cudaDeviceSynchronize();
        cudaError = cudaGetLastError();
        if (cudaError != cudaSuccess)
        {
            throw runtime_error("filterKernel failed: " + string(cudaGetErrorString(cudaError)));
        }
        param.timeMsStep4Filter = timerLocal.tocMs();

        // Step5 - copy eligible pairs
        timerLocal.tic();
        ReqDocPair *d_endPtr = thrust::copy_if(thrust::device,
                                            param.bufData.d_data,
                                            param.bufData.d_data + param.activePairSize,
                                            param.rstData.d_data,
                                            NonZeroPredicator());
        param.activePairSize = d_endPtr - param.rstData.d_data;
        param.timeMsStep5CopyIf = timerLocal.tocMs();

        // Step6 - scoring
        timerLocal.tic();
        gridSize = (param.activePairSize + blockSize - 1) / blockSize;
        scoringKernel<<<gridSize, blockSize>>>(param);
        cudaDeviceSynchronize();
        cudaError = cudaGetLastError();
        if (cudaError != cudaSuccess)
        {
            throw runtime_error("scoringKernel failed (2): " + string(cudaGetErrorString(cudaError)));
        }
        param.timeMsStep6Scoring = timerLocal.tocMs();

        // Step7 - topk exact
        topkParam.numReqDocPairs = param.activePairSize;
        topkParam.numToRetrieve = param.numToRetrieve;
        vector<ReqDocPair> rst = param.topk.retrieveTopk(topkParam);
        param.timeMsStep7s1TopkExactUpdateCounter = topkParam.timeMsUpdateCounter;
        param.timeMsStep7s2TopkExactCopyCounterToCpu = topkParam.timeMsCopyCounterToCpu;
        param.timeMsStep7s3TopkExactFindLowestBucket = topkParam.timeMsFindLowestBucket;
        param.timeMsStep7s4TopkExactPrefilter = topkParam.timeMsPrefilter;
        param.timeMsStep7s5TopkExactSort = topkParam.timeMsSort;
        param.timeMsStep7s6TopkExactCopyBackToCpu = topkParam.timeMsCopyBackToCpu;
        param.timeMsStep7TopkExactTotal = topkParam.timeMsTotal;

        param.timeMsTotal = timerGlobal.tocMs();
        
        return rst;
    }

}

vector<vector<ReqDocPair>> postGpuAlgoBatch(const vector<CentroidCpu> &centroids,
                                            const vector<ItemCpu> &reqs,
                                            const vector<ItemCpu> &docs,
                                            int numToRetrieve,
                                            bool enableBidAware,
                                            int minScore,
                                            int maxScore)
{
    assert(minScore < maxScore);

    // prepare PostGpuAlgoParam
    PostGpuAlgoParam param;
    param.docData.init(docs);
    param.numToRetrieve = numToRetrieve;
    param.topk.init(minScore, maxScore);
    param.reqData.init(reqs);
    param.centroidData.init(centroids);
    param.rstData.malloc(docs.size());
    param.bufData.malloc(docs.size());
    param.enableBidAware = enableBidAware;
    cudaError_t cudaError = cudaMallocManaged(&param.d_centroidScore, param.centroidData.size_d_centroidId * sizeof(float));
    if (cudaError != cudaSuccess)
    {
        throw runtime_error("cudaMallocManaged failed (param.d_centroidScore): " + string(cudaGetErrorString(cudaError)));
    }
    
    vector<vector<ReqDocPair>> rst2D;

    vector<float> timeMsStep1ScoreCentroid1D;
    vector<float> timeMsStep2PreScore1D;
    vector<float> timeMsStep3s1TopkApproxUpdateCounter1D;
    vector<float> timeMsStep3s2TopkApproxCopyCounterToCpu1D;
    vector<float> timeMsStep3s3TopkApproxFindLowestBucket1D;
    vector<float> timeMsStep3s4TopkApproxPrefilter1D;
    vector<float> timeMsStep3s5TopkApproxCopyback1D;
    vector<float> timeMsStep3TopkApproxTotal1D;
    vector<float> timeMsStep4Filter1D;
    vector<float> timeMsStep5CopyIf1D;
    vector<float> timeMsStep6Scoring1D;
    vector<float> timeMsStep7s1TopkExactUpdateCounter1D;
    vector<float> timeMsStep7s2TopkExactCopyCounterToCpu1D;
    vector<float> timeMsStep7s3TopkExactFindLowestBucket1D;
    vector<float> timeMsStep7s4TopkExactPrefilter1D;
    vector<float> timeMsStep7s5TopkExactSort1D;
    vector<float> timeMsStep7s6TopkExactCopyBackToCpu1D;
    vector<float> timeMsStep7TopkExactTotal1D;
    vector<float> timeMsTotal1D;


    for (int reqIdx = 0; reqIdx < reqs.size(); reqIdx++)
    {
        if ((reqIdx+1) % 500 == 0)
        {
            cout << "Processing request " << reqIdx+1 << "..." << endl;
        }

        param.reqIdx = reqIdx;
        rst2D.push_back(postGpuAlgoSingle(param));

        timeMsStep1ScoreCentroid1D.push_back(param.timeMsStep1ScoreCentroid);
        timeMsStep2PreScore1D.push_back(param.timeMsStep2PreScore);
        timeMsStep3s1TopkApproxUpdateCounter1D.push_back(param.timeMsStep3s1TopkApproxUpdateCounter);
        timeMsStep3s2TopkApproxCopyCounterToCpu1D.push_back(param.timeMsStep3s2TopkApproxCopyCounterToCpu);
        timeMsStep3s3TopkApproxFindLowestBucket1D.push_back(param.timeMsStep3s3TopkApproxFindLowestBucket);
        timeMsStep3s4TopkApproxPrefilter1D.push_back(param.timeMsStep3s4TopkApproxPrefilter);
        timeMsStep3s5TopkApproxCopyback1D.push_back(param.timeMsStep3s5TopkApproxCopyback);
        timeMsStep3TopkApproxTotal1D.push_back(param.timeMsStep3TopkApproxTotal);
        timeMsStep4Filter1D.push_back(param.timeMsStep4Filter);
        timeMsStep5CopyIf1D.push_back(param.timeMsStep5CopyIf);
        timeMsStep6Scoring1D.push_back(param.timeMsStep6Scoring);
        timeMsStep7s1TopkExactUpdateCounter1D.push_back(param.timeMsStep7s1TopkExactUpdateCounter);
        timeMsStep7s2TopkExactCopyCounterToCpu1D.push_back(param.timeMsStep7s2TopkExactCopyCounterToCpu);
        timeMsStep7s3TopkExactFindLowestBucket1D.push_back(param.timeMsStep7s3TopkExactFindLowestBucket);
        timeMsStep7s4TopkExactPrefilter1D.push_back(param.timeMsStep7s4TopkExactPrefilter);
        timeMsStep7s5TopkExactSort1D.push_back(param.timeMsStep7s5TopkExactSort);
        timeMsStep7s6TopkExactCopyBackToCpu1D.push_back(param.timeMsStep7s6TopkExactCopyBackToCpu);
        timeMsStep7TopkExactTotal1D.push_back(param.timeMsStep7TopkExactTotal);
        timeMsTotal1D.push_back(param.timeMsTotal);

    }
    param.reqData.reset();
    param.docData.reset();
    param.centroidData.reset();
    cudaFree(param.d_centroidScore);

    printLatency(timeMsStep1ScoreCentroid1D, "Step1 ScoreCentroid");
    printLatency(timeMsStep2PreScore1D, "Step2 PreScore");
    printLatency(timeMsStep3s1TopkApproxUpdateCounter1D, "Step3s1 TopkApproxUpdateCounter");
    printLatency(timeMsStep3s2TopkApproxCopyCounterToCpu1D, "Step3s2 TopkApproxCopyCounterToCpu");
    printLatency(timeMsStep3s3TopkApproxFindLowestBucket1D, "Step3s3 TopkApproxFindLowestBucket");
    printLatency(timeMsStep3s4TopkApproxPrefilter1D, "Step3s4 TopkApproxPrefilter");
    printLatency(timeMsStep3s5TopkApproxCopyback1D, "Step3s5 TopkApproxCopyback");
    printLatency(timeMsStep3TopkApproxTotal1D, "Step3 TopkApproxTotal");
    printLatency(timeMsStep4Filter1D, "Step4 Filter");
    printLatency(timeMsStep5CopyIf1D, "Step5 CopyIf");
    printLatency(timeMsStep6Scoring1D, "Step6 Scoring");
    printLatency(timeMsStep7s1TopkExactUpdateCounter1D, "Step7s1 TopkExactUpdateCounter");
    printLatency(timeMsStep7s2TopkExactCopyCounterToCpu1D, "Step7s2 TopkExactCopyCounterToCpu");
    printLatency(timeMsStep7s3TopkExactFindLowestBucket1D, "Step7s3 TopkExactFindLowestBucket");
    printLatency(timeMsStep7s4TopkExactPrefilter1D, "Step7s4 TopkExactPrefilter");
    printLatency(timeMsStep7s5TopkExactSort1D, "Step7s5 TopkExactSort");
    printLatency(timeMsStep7s6TopkExactCopyBackToCpu1D, "Step7s6 TopkExactCopyBackToCpu");
    printLatency(timeMsStep7TopkExactTotal1D, "Step7 TopkExactTotal");
    printLatency(timeMsTotal1D, "Total");

    return rst2D;
}
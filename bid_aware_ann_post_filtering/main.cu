#include "data_synthesizer.cuh"
#include "pre_cpu.cuh"
#include "pre_gpu.cuh"
#include "eval.cuh"
#include "common.cuh"
#include "topk.cuh"

#include <iostream>
#include <cassert>

using namespace std;

int main()
{
    const int kNumCentroids = 10;
    const int kDim = 4;
    const float kCentroidStdDev = 0.1;
    const float kDocStdDev = 0.05;
    const float kPassRate = 0.25;
    const int kBidStdDev = 1;
    const int kNumDocsPerCentroid = 1000;
    const int kNumReqsPerCentroid = 1;
    const int kNumToRetrieve = 100;

    // prepare CPU data
    vector<CentroidCpu> centroids = genRandCentroids(kNumCentroids, kDim, kCentroidStdDev);
    vector<ItemCpu> docs = genRandReqDocsFromCentroids(centroids, kDocStdDev, kNumDocsPerCentroid, kBidStdDev);
    vector<ItemCpu> reqs = genRandReqDocsFromCentroids(centroids, kDocStdDev, kNumReqsPerCentroid, kBidStdDev);
    for (auto &req : reqs)
    {
        req.randAttr = kPassRate;
    }

    vector<vector<ReqDocPair>> rstPreCpu = preCpuAlgoBatch(reqs, docs, kNumToRetrieve);
    vector<vector<ReqDocPair>> rstPreGpu = preGpuAlgoBatch(reqs, docs, kNumToRetrieve);

    for (int i = 0; i < reqs.size(); i++)
    {
        float sameClusterRatio = checkSameClusterRatio(rstPreCpu[i]);
        cout << "Same cluster ratio: " << sameClusterRatio << endl;
        float sameClusterRatioPreGpu = checkSameClusterRatio(rstPreGpu[i]);
        cout << "Same cluster ratio: " << sameClusterRatioPreGpu << endl;
    }

    return 0;
}
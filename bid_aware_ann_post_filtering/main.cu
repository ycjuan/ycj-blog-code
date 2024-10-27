#include "data_synthesizer.cuh"
#include "pre_cpu.cuh"
#include "pre_gpu.cuh"
#include "eval.cuh"
#include "common.cuh"
#include "topk.cuh"
#include "config.cuh"

#include <iostream>
#include <cassert>

using namespace std;

int main()
{

    // prepare CPU data
    cout << "Generating data..." << endl;
    vector<CentroidCpu> centroids = genRandCentroids(kNumCentroids, kDim, kCentroidStdDev);
    vector<ItemCpu> docs = genRandReqDocsFromCentroids(centroids, kDocStdDev, kNumDocsPerCentroid, kBidStdDev);
    vector<ItemCpu> reqs = genRandReqDocsFromCentroids(centroids, kDocStdDev, kNumReqsPerCentroid, kBidStdDev);
    for (auto &req : reqs)
    {
        req.randAttr = kPassRate;
    }

    cout << "Running pre CPU algo..." << endl;
    vector<vector<ReqDocPair>> rstPreCpu = preCpuAlgoBatch(reqs, docs, kNumToRetrieve);

    cout << "Running pre GPU algo..." << endl;
    vector<vector<ReqDocPair>> rstPreGpu = preGpuAlgoBatch(reqs, docs, kNumToRetrieve);

    cout << "Comparing results..." << endl;
    double sameClusterRatioSum = 0;
    for (int reqIdx = 0; reqIdx < reqs.size(); reqIdx++)
    {
        int numErrors = compareResults(rstPreCpu[reqIdx], rstPreGpu[reqIdx]);
        if (numErrors > 4)
        {
            cout << "Error in comparing results: " << numErrors << endl;
        }
        float sameClusterRatio = checkSameClusterRatio(rstPreCpu[reqIdx]);
        sameClusterRatioSum += sameClusterRatio;
    }
    double avgSameClusterRatio = sameClusterRatioSum / reqs.size();
    cout << "Average same cluster ratio: " << avgSameClusterRatio << endl;

    return 0;
}
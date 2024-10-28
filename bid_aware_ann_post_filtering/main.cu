#include "data_synthesizer.cuh"
#include "pre_cpu.cuh"
#include "pre_gpu.cuh"
#include "post_gpu.cuh"
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
    float minScore;
    float maxScore;
    getUpperAndLowerBound(reqs, docs, minScore, maxScore);

    cout << endl << "Running pre CPU algo..." << endl;
    vector<vector<ReqDocPair>> rstPreCpu = preCpuAlgoBatch(reqs, docs, kNumToRetrieve);

    cout << endl << "Running pre GPU algo..." << endl;
    vector<vector<ReqDocPair>> rstPreGpu = preGpuAlgoBatch(reqs, docs, kNumToRetrieve, minScore, maxScore);

    cout << endl << "Running post GPU algo (bid aware)..." << endl;
    bool enableBidAware = true;
    vector<vector<ReqDocPair>> rstPostGpuBidAware = postGpuAlgoBatch(centroids, reqs, docs, kNumToRetrieve, enableBidAware, minScore, maxScore);

    cout << endl << "Running post GPU algo (no bid aware)..." << endl;
    enableBidAware = false;
    vector<vector<ReqDocPair>> rstPostGpuNoBidAware = postGpuAlgoBatch(centroids, reqs, docs, kNumToRetrieve, enableBidAware, minScore, maxScore);

    cout << endl << "Comparing results..." << endl;
    double sameClusterRatioSum = 0;
    double recallBidAwareSum = 0;
    double recallNoBidAwareSum = 0;
    for (int reqIdx = 0; reqIdx < reqs.size(); reqIdx++)
    {
        if (kRunCpu)
        {
            int numErrors = compareResults(rstPreGpu[reqIdx], rstPreCpu[reqIdx]);
            if (numErrors > 4)
            {
                cout << "Error in comparing results: " << numErrors << endl;
            }
        }
        float sameClusterRatio = checkSameClusterRatio(rstPreGpu[reqIdx]);
        sameClusterRatioSum += sameClusterRatio;

        float recall = checkRecall(rstPostGpuBidAware[reqIdx], rstPreGpu[reqIdx]);
        recallBidAwareSum += recall;

        recall = checkRecall(rstPostGpuNoBidAware[reqIdx], rstPreGpu[reqIdx]);
        recallNoBidAwareSum += recall;
    }
    double avgSameClusterRatio = sameClusterRatioSum / reqs.size();
    double avgRecallBidAware = recallBidAwareSum / reqs.size();
    double avgRecallNoBidAware = recallNoBidAwareSum / reqs.size();
    cout << "Average same cluster ratio: " << avgSameClusterRatio << endl;
    cout << "Average recall (bid aware): " << avgRecallBidAware << endl;
    cout << "Average recall (no bid aware): " << avgRecallNoBidAware << endl;

    return 0;
}
#ifndef PRE_CPU_CUH
#define PRE_CPU_CUH

#include "data_synthesizer.cuh"
#include "common.cuh"
#include "util.cuh"

#include <random>
#include <iostream>
#include <algorithm>


using namespace std;

vector<ReqDocPair> preCpuAlgoSingle(const ItemCpu &req, const vector<ItemCpu> &docs, int k)
{
    vector<ReqDocPair> rst;
    for (auto doc : docs)
    {
        if (doc.randAttr <= req.randAttr)
        {
            float score = getScore(req, doc) * doc.bid;
            
            ReqDocPair pair;
            pair.reqIdx = req.uid;
            pair.docIdx = doc.uid;
            pair.score = score;
            pair.reqCentroidId = req.centroidId;
            pair.docCentroidId = doc.centroidId;

            rst.push_back(pair);
        }
    }

    sort(rst.begin(), rst.end(), scoreComparator);

    if (rst.size() > k)
    {
        rst.resize(k);
    }

    return rst;
}

vector<vector<ReqDocPair>> preCpuAlgoBatch(const vector<ItemCpu> &reqs, const vector<ItemCpu> &docs, int k)
{
    vector<vector<ReqDocPair>> rst2D(reqs.size());
    CudaTimer timer;
    timer.tic();
    #pragma omp parallel for
    for (int reqIdx = 0; reqIdx < reqs.size(); reqIdx++)
    {
        rst2D[reqIdx] = preCpuAlgoSingle(reqs[reqIdx], docs, k);
    }
    float timeMs = timer.tocMs() / reqs.size();
    cout << "Latency per request (CPU): " << timeMs << " ms" << endl;

    return rst2D;
}

#endif // PRE_CPU_CUH
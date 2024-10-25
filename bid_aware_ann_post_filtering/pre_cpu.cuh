#ifndef PRE_CPU_CUH
#define PRE_CPU_CUH

#include "data_synthesizer.cuh"
#include "common.cuh"

#include <random>
#include <iostream>
#include <algorithm>


using namespace std;

vector<ReqDocPair> preCpuAlgoSingle(const ItemCpu &req, const vector<ItemCpu> &docs, int k)
{
    vector<ReqDocPair> eligibleDocs;
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

            eligibleDocs.push_back(pair);
        }
    }

    sort(eligibleDocs.begin(), eligibleDocs.end(), scoreComparator);

    if (eligibleDocs.size() > k)
    {
        eligibleDocs.resize(k);
    }

    return eligibleDocs;
}

vector<vector<ReqDocPair>> preCpuAlgoBatch(const vector<ItemCpu> &reqs, const vector<ItemCpu> &docs, int k)
{
    vector<vector<ReqDocPair>> eligibleDocsBatch;
    for (auto req : reqs)
    {
        auto eligibleDocs = preCpuAlgoSingle(req, docs, k);
        eligibleDocsBatch.push_back(eligibleDocs);
    }

    return eligibleDocsBatch;
}

#endif // PRE_CPU_CUH
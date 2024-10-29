#ifndef EVAL_CUH
#define EVAL_CUH

#include "data_struct.cuh"

#include <vector>
#include <set>

using namespace std;

float checkSameClusterRatio(const vector<ReqDocPair> &pair1D)
{
    int sameClusterCount = 0;
    for (const auto &rq : pair1D)
    {
        if (rq.reqCentroidId == rq.docCentroidId)
        {
            sameClusterCount++;
        }
    }

    return (float)sameClusterCount / pair1D.size();
}

int compareResults(const vector<ReqDocPair> &pair1D, const vector<ReqDocPair> &pair1DRef)
{
    if (pair1D.size() != pair1DRef.size())
    {
        return -1;
    }

    int errorCount = 0;
    for (int i = 0; i < pair1D.size(); i++)
    {
        if (pair1D[i].reqIdx != pair1DRef[i].reqIdx ||
            pair1D[i].docIdx != pair1DRef[i].docIdx ||
            pair1D[i].score != pair1DRef[i].score)
        {
            errorCount++;
        }
    }

    return errorCount;
}

float checkRecall(const vector<ReqDocPair> &pair1D, const vector<ReqDocPair> &pair1DRef)
{
    if (pair1D.size() != pair1DRef.size())
    {
        throw runtime_error("pair1D.size() != pair1DRef.size()");
    }

    set<int> docIdxSet;
    for (const auto &rd : pair1DRef)
    {
        docIdxSet.insert(rd.docIdx);
    }

    int recallCount = 0;
    for (const auto &rd : pair1D)
    {
        if (docIdxSet.find(rd.docIdx) != docIdxSet.end())
        {
            recallCount++;
        }
    }
    float recall = (float)recallCount / docIdxSet.size();

    return recall;
}

/*
float checkSizeToCover(const vector<ReqDocPair> &pair1D, const vector<ReqDocPair> &pair1DRef)
{
    if (pair1D.size() != pair1DRef.size())
    {
        throw runtime_error("pair1D.size() != pair1DRef.size()");
    }

    set<int> docIdxSet;
    for (const auto &rd : pair1DRef)
    {
        docIdxSet.insert(rd.docIdx);
    }

    int sizeToCover = 0;
    set<int> coveredDocIdxSet;
    for (const auto &rd : pair1D)
    {
        if (docIdxSet.find(rd.docIdx) != docIdxSet.end() &&
            coveredDocIdxSet.find(rd.docIdx) == coveredDocIdxSet.end())
        {
            coveredDocIdxSet.insert(rd.docIdx);
            sizeToCover++;
        }
    }

    return (float)sizeToCover / docIdxSet.size();
}
*/

#endif // EVAL_CUH
#ifndef MISC_CUH
#define MISC_CUH

#include "data_struct.cuh"
#include "common.cuh"

#include <vector>
#include <numeric>
#include <algorithm>
#include <random>

using namespace std;

inline void getUpperAndLowerBound(const std::vector<ItemCpu> &req1D,
                                  const std::vector<ItemCpu> &doc1D,
                                  float &minScore,
                                  float &maxScore)
{
    int kMaxNumReqs = 2000;
    int kMaxNumDocs = 50000;

    vector<int> reqIdx1D(req1D.size());
    for (int i = 0; i < req1D.size(); i++)
    {
        reqIdx1D[i] = req1D[i].uid;
    }

    vector<int> docIdx1D(doc1D.size());
    for (int i = 0; i < doc1D.size(); i++)
    {
        docIdx1D[i] = doc1D[i].uid;
    }

    shuffle(reqIdx1D.begin(), reqIdx1D.end(), default_random_engine(0));
    shuffle(docIdx1D.begin(), docIdx1D.end(), default_random_engine(0));

    if (reqIdx1D.size() > kMaxNumReqs)
    {
        reqIdx1D.resize(kMaxNumReqs);
    }

    if (docIdx1D.size() > kMaxNumDocs)
    {
        docIdx1D.resize(kMaxNumDocs);
    }

    sort(reqIdx1D.begin(), reqIdx1D.end());
    sort(docIdx1D.begin(), docIdx1D.end());

    vector<float> scores(reqIdx1D.size() * docIdx1D.size());
    #pragma omp parallel for
    for (int i = 0; i < reqIdx1D.size(); i++)
    {
        for (int j = 0; j < docIdx1D.size(); j++)
        {
            int reqIdx = reqIdx1D[i];
            int docIdx = docIdx1D[j];
            scores[i * docIdx1D.size() + j] = getScore(req1D[reqIdx], doc1D[docIdx]) * doc1D[docIdx].bid;
        }
    }

    sort(scores.begin(), scores.end());

    minScore = scores[(int)(scores.size() * 0.002)];
    maxScore = scores[(int)(scores.size() * 0.998)];

    cout << "minScore: " << minScore << ", maxScore: " << maxScore << endl;
}

inline void printLatency(const vector<float> &latency1D, const string &prefix)
{
    float latencySum = accumulate(latency1D.begin(), latency1D.end(), 0.0);
    float latencyAvg = latencySum / latency1D.size();

    cout << prefix << " latencyAvg: " << latencyAvg << " ms" << endl;
}

#endif // MISC_CUH
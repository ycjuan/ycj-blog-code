#ifndef MISC_CUH
#define MISC_CUH

#include "data_struct.cuh"
#include "common.cuh"

#include <vector>
#include <numeric>
#include <algorithm>

using namespace std;

inline void getUpperAndLowerBound(const std::vector<ItemCpu> &req1D,
                                  const std::vector<ItemCpu> &doc1D,
                                  float &minScore,
                                  float &maxScore)
{
    vector<float> scores;
    for (auto req : req1D)
    {
        for (auto doc : doc1D)
        {
            scores.push_back(getScore(req, doc));
        }
    }

    sort(scores.begin(), scores.end());

    minScore = scores[(int)(scores.size() * 0.002)];
    maxScore = scores[(int)(scores.size() * 0.998)];
}

inline void printLatency(const vector<float> &latency1D, const string &prefix)
{
    float latencySum = accumulate(latency1D.begin(), latency1D.end(), 0.0);
    float latencyAvg = latencySum / latency1D.size();

    cout << prefix << " latencyAvg: " << latencyAvg << " ms" << endl;
}

#endif // MISC_CUH
#ifndef MISC_CUH
#define MISC_CUH

#include "data_struct.cuh"
#include "common.cuh"

#include <vector>
#include <algorithm>

void getUpperAndLowerBound(const std::vector<ItemCpu> &req1D,
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

#endif // MISC_CUH
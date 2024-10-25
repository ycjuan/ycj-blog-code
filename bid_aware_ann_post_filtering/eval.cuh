#ifndef EVAL_CUH
#define EVAL_CUH

#include "data_struct.cuh"

#include <vector>

float checkSameClusterRatio(const std::vector<ReqDocPair> &rq1D)
{
    int sameClusterCount = 0;
    for (const auto &rq : rq1D)
    {
        if (rq.reqCentroidId == rq.docCentroidId)
        {
            sameClusterCount++;
        }
    }

    return (float)sameClusterCount / rq1D.size();
}

#endif // EVAL_CUH
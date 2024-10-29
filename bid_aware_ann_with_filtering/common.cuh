#ifndef COMMON_CUH
#define COMMON_CUH

#include "data_struct.cuh"
#include "topk.cuh"

using namespace std;

inline __device__ __host__ bool scoreComparator(const ReqDocPair &a, const ReqDocPair &b)
{
    return a.score > b.score;
}

struct ScorePredicator
{
    inline __host__ __device__ bool operator()(const ReqDocPair &a, const ReqDocPair &b)
    {
        return scoreComparator(a, b);
    }
};

inline float getScore(const ItemCpu &a, const ItemCpu &b)
{
    double dist = 0.0f;
    for (int j = 0; j < a.emb.size(); j++)
    {
        dist += (a.emb[j] - b.emb[j]) * (a.emb[j] - b.emb[j]);
    }
    dist = sqrt(dist);
    return 1.0 / (dist + 1e-8);
};

inline __device__ float getScoreDevice(
    const ItemDataGpu &reqData, const ItemDataGpu &docData, int reqIdx, int docIdx)
{
    double dist = 0.0f;
    for (int j = 0; j < reqData.embDim; j++)
    {
        float a = reqData.getEmb(reqIdx, j);
        float b = docData.getEmb(docIdx, j);
        dist += (a - b) * (a - b);
    }
    dist = sqrt(dist);
    return 1.0 / (dist + 1e-8);
};

inline __device__ float getScoreDevice(
    const ItemDataGpu &reqData, const CentroidDataGpu &centroidData, int reqIdx, int centroidIdx)
{
    double dist = 0.0f;
    for (int j = 0; j < reqData.embDim; j++)
    {
        float a = reqData.getEmb(reqIdx, j);
        float b = centroidData.getEmb(centroidIdx, j);
        dist += (a - b) * (a - b);
    }
    dist = sqrt(dist);
    return 1.0 / (dist + 1e-8);
};

#endif // COMMON_CUH
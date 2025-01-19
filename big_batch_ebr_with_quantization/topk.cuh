#ifndef TOPK_CUH
#define TOPK_CUH

#include <vector>

#include "common.cuh"
#include "thrust_allocator.cuh"

struct TopkParam
{
    float *d_score;
    float *h_score;
    Pair *h_rstCpu;
    Pair *d_rstGpu;
    int numReqs;
    int numDocs;
    int numToRetrieve;
    bool useRandomSampling = false;
    float cpuTimeMs = 0;
    float gpuSampleTimeMs = 0;
    float gpuFindThresholdTimeMs = 0;
    float gpuCopyEligibleTimeMs = 0;
    float gpuRetreiveExactTimeMs = 0;
    float gpuTotalTimeMs = 0;
    float gpuApproxTimeMs = 0;
};

inline __device__ __host__ bool scoreComparator(const Pair &a, const Pair &b)
{
    return a.score > b.score;
}

struct ScorePredicator
{
    inline __host__ __device__ bool operator()(const Pair &a, const Pair &b)
    {
        return scoreComparator(a, b);
    }
};

inline __device__ __host__ size_t getMemAddr(int reqIdx, int docIdx, size_t numDocs)
{
    return reqIdx * numDocs + docIdx;
}

class TopkSampling
{
public:
    void malloc();

    void free();

    void retrieveTopk(TopkParam &param);

private:

    // constants
    const size_t kNumSamplesPerReq = 10000;
    const size_t kMaxNumReqs = 1 << 9;
    const size_t kMaxEligiblePairsPerReq = 1 << 18;

    // device memory
    float *d_scoreSample = nullptr;
    float *d_scoreThreshold = nullptr;
    Pair *d_eligiblePairs = nullptr;
    int *d_copyCount = nullptr;
    int *h_copyCount = nullptr;

    // functions
    void sample(TopkParam &param);
    void findThreshold(TopkParam &param);
    void copyEligible(TopkParam &param);
    void retrieveExact(TopkParam &param);

    // others
    StaticThrustAllocator thrustAllocator;
};

void retrieveTopkCpu(TopkParam &param);

#endif // TOPK_CUH
#ifndef TOPK_CUH
#define TOPK_CUH

#include <vector>

#include "common.cuh"
#include "thrust_allocator.cuh"

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
    float *dm_scoreThreshold = nullptr;
    Pair *d_eligiblePairs = nullptr;
    int *dm_copyCount = nullptr;

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
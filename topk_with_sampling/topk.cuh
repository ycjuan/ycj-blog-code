#ifndef TOPK_CUH
#define TOPK_CUH

#include <vector>

#include "common.cuh"

class TopkSampling
{
public:
    void init();

    void reset();

    void retrieveTopk(TopkParam &param);

private:

    const size_t kNumSamplesPerReq = 10000;

    const size_t kMaxNumReqs = 1 << 10;

    float *dm_scoreSample = nullptr;

    void sample(TopkParam &param);

    void findThreshold(TopkParam &param, float &threshold);

    void copyEligible(TopkParam &param, float threshold, size_t numCopied);

    void retrieveExact(TopkParam &param);
};

void retrieveTopkCpu(TopkParam &param);

#endif // TOPK_CUH
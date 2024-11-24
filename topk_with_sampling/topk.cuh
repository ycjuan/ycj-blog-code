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
    float *dm_scoreSample = nullptr;

    void sample(TopkParam &param);

    void findThreshold(TopkParam &param, float &threshold);
};

void retrieveTopkCpu(TopkParam &param);

#endif // TOPK_CUH
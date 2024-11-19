#ifndef COMMON_CUH
#define COMMON_CUH

#include <vector>
#include <string>

enum CopyMode
{
    MEMCPY = 0,
    FOR_LOOP = 1,
    DISK = 2
};

struct ExpSetting
{
    long numTrials;
    long numDocsAll;
    long numDocsSelected;
    long numDims;
    CopyMode copyMode;
    std::string binaryPath;
    bool hasGpu;
    long numThreads;
};

struct ExpData
{
    std::vector<float> hv_embAll;
    std::vector<long> hv_docIds;
    float *d_embSelected = nullptr; 
    float *hp_embSelected = nullptr;

    ~ExpData()
    {
        if (d_embSelected != nullptr)
            cudaFree(d_embSelected);
        if (hp_embSelected != nullptr)
            cudaFreeHost(hp_embSelected);
    }
};

#endif // COMMON_CUH

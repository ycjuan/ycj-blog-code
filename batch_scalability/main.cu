#include "data.cuh"
#include "methods.cuh"
#include "util.cuh"
#include <sstream>
#include <stdexcept>
#include <iostream>

using namespace BatchScalability;

void compareResult(Data& data)
{
    for (int reqIdx = 0; reqIdx < data.numReqs; reqIdx++)
    {
        for (int docIdx = 0; docIdx < data.numDocs; docIdx++)
        {
            float cpuVal = data.h_rstDataCpu[getMemAddrRst(reqIdx, docIdx, data.numReqs, data.numDocs)];
            float gpuVal = data.d_rstDataGpu[getMemAddrRst(reqIdx, docIdx, data.numReqs, data.numDocs)];
            if (abs(cpuVal - gpuVal) / abs(gpuVal) > 1e-3)
            {
                std::ostringstream oss;
                oss << "Mismatch at (" << reqIdx << ", " << docIdx << "): " << cpuVal << " != " << gpuVal << std::endl;
                throw std::runtime_error(oss.str());
            }
        }
    }

    std::cout << "All results are correct" << std::endl;
}

int main()
{
    const int kNumReqs = 16;
    const int kNumDocs = 1000000;
    const int kEmbDim= 128;
    const int kNumTrials = 100;

    Data data = genData(kNumReqs, kNumDocs, kEmbDim);

    methodCpu(data);

    {
        Timer timer;
        timer.tic();
        for (int t = 0; t < kNumTrials; t++)
        {
            methodGpuNaive(data);
        }
        float timeMs = timer.tocMs() / kNumTrials;
        std::cout << "GPU naive time: " << timeMs << " ms" << std::endl;
    }

    compareResult(data);

    freeData(data);

    return 0;
}
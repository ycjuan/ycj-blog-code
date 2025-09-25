#include "data.cuh"
#include "methods.cuh"
#include "util.cuh"
#include <sstream>
#include <stdexcept>
#include <iostream>
#include <functional>

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

void runExp(Data& data, std::function<void(Data&)> method, const std::string& methodName, int numTrials = 100)
{
    Timer timer;
    timer.tic();
    for (int t = 0; t < numTrials; t++)
    {
        method(data);
    }
    float timeMs = timer.tocMs() / numTrials;
    compareResult(data);
    std::cout << methodName << " time: " << timeMs << " ms" << std::endl;
}


int main()
{
    const int kNumReqs = 16;
    const int kNumDocs = 1000000;
    const int kEmbDim= 128;
    const int kNumTrials = 100;

    Data data = genData(kNumReqs, kNumDocs, kEmbDim);

    methodCpu(data);

    // Using function pointers with std::function
    runExp(data, methodGpuNaive1, "GPU naive 1", kNumTrials);
    runExp(data, methodGpuNaive2, "GPU naive 2", kNumTrials);

    freeData(data);

    return 0;
}
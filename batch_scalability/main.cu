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
    CHECK_CUDA(cudaMemset(data.d_rstDataGpu, 0, data.numReqs * data.numDocs * sizeof(float)));
    Timer timer;
    for (int t = -3; t < numTrials; t++)
    {
        if (t == 0)
            timer.tic();
        method(data);
    }
    float timeMs = timer.tocMs() / numTrials;
    compareResult(data);
    std::cout << methodName << " time: " << timeMs << " ms" << std::endl;
}


int main()
{
    printDeviceInfo();

    const int kNumReqs = 16;
    const int kNumDocs = 1024 * 1024;
    const int kEmbDim = 128;
    const int kNumTrials = 10;

    Data data = genData(kNumReqs, kNumDocs, kEmbDim);

    methodCpu(data);

    //runExp(data, methodGpu1, "GPU 1", kNumTrials);
    //runExp(data, methodGpu2, "GPU 2", kNumTrials);
    //runExp(data, methodGpu3, "GPU 3", kNumTrials);
    runExp(data, methodGpu4, "GPU 4", kNumTrials);
    //runExp(data, methodGpu5, "GPU 5", kNumTrials);
    runExp(data, methodGpu6, "GPU 6", kNumTrials);

    freeData(data);

    return 0;
}
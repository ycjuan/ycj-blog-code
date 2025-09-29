#include "data.cuh"
#include "methods.cuh"
#include "util.cuh"
#include <sstream>
#include <stdexcept>
#include <iostream>
#include <functional>

using namespace MatMatMulFromScratch;

void compareResult(Data& data)
{
    for (int reqIdx = 0; reqIdx < data.M; reqIdx++)
    {
        for (int docIdx = 0; docIdx < data.N; docIdx++)
        {
            float cpuVal = data.h_C[getMemAddrC(reqIdx, docIdx, data.M, data.N)];
            float gpuVal = data.d_C[getMemAddrC(reqIdx, docIdx, data.M, data.N)];
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
    CHECK_CUDA(cudaMemset(data.d_C, 0, data.M * data.N * sizeof(float)));
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

    //runExp(data, methodCublas, "CUBLAS", kNumTrials);
    
    freeData(data);

    return 0;
}
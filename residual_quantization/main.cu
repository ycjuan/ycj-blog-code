#include <iostream>
#include <stdexcept>
#include <vector>

#include "data.cuh"
#include "methods.cuh"
#include "util.cuh"

std::vector<EMB_T> copyResults(Data data)
{
    std::vector<EMB_T> rst(data.config.numToScore * data.config.embDim);
    CHECK_CUDA(cudaMemcpy(rst.data(), data.d_rst, data.config.numToScore * data.config.embDim * sizeof(EMB_T),
                          cudaMemcpyDeviceToHost));
    return rst;
}

float computeRMSE(std::vector<EMB_T> rstA, std::vector<EMB_T> rstB, int numToScore, int embDim)
{
    if (rstA.size() != rstB.size())
    {
        throw std::runtime_error("rstA.size() != rstB.size()");
    }

    double sum = 0;
    #pragma omp parallel for reduction(+:sum)
    for (int i = 0; i < numToScore; i++)
    {
        double diffSum = 0;
        double twoNormSum = 0;
        for (int j = 0; j < embDim; j++)
        {
            size_t memAddr = getMemAddr(i, j, numToScore, embDim);
            float valA = static_cast<float>(rstA[memAddr]);
            float valB = static_cast<float>(rstB[memAddr]);
            float diff = (valA - valB);
            diffSum += diff * diff;
            twoNormSum += valA * valA;
        }
        float twoNorm = sqrt(twoNormSum / embDim);
        float rmse = sqrt(diffSum / embDim);
        float normalizedRMSE = rmse / twoNorm;
        sum += normalizedRMSE;
    }
    return sum / numToScore;
}

void runExp(Data data, Method method, const std::string& methodName, std::vector<EMB_T> rstRef, int numTrials = 10)
{
    CHECK_CUDA(cudaMemset(data.d_rst, 0, data.config.numToScore * data.config.embDim * sizeof(EMB_T)));

    Timer timer;
    for (int t = -3; t < numTrials; t++)
    {
        if (t == 0)
        {
            timer.tic();
        }
        runMethod(data, method);
    }
    float timeMs = timer.tocMs() / numTrials;
    timeMs /= numTrials;
    std::cout << methodName << " time: " << timeMs << " ms" << std::endl;

    std::vector<EMB_T> rst = copyResults(data);
    float rmse = computeRMSE(rstRef, rst, data.config.numToScore, data.config.embDim);
    std::cout << methodName << " RMSE: " << rmse << std::endl;
}

int main()
{
    int kNumTrials = 10;

    Config config;
    config.numDocs = 1000000;
    config.numToScore = 10000;
    config.embDim = 4096;
    config.numBitsPerDim = 2;
    config.numCentroids = 10000;
    config.stdDev = 0.1f;
    config.debugMode = false;
    config.validate();

    Data data = genData(config);

    runMethod(data, Method::REFERENCE);
    std::vector<EMB_T> rstRef = copyResults(data);
    runExp(data, Method::REFERENCE, "Reference", rstRef, kNumTrials);
    runExp(data, Method::BASELINE_H2D, "Baseline H2D", rstRef, kNumTrials);
    runExp(data, Method::BASELINE_D2D, "Baseline D2D", rstRef, kNumTrials);
    runExp(data, Method::RES_QUANT_H2D, "Residual Quant H2D", rstRef, kNumTrials);
    runExp(data, Method::RES_QUANT_D2D, "Residual Quant D2D", rstRef, kNumTrials);

    return 0;
}
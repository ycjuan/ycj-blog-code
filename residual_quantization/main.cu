#include <iostream>
#include <random>
#include <stdexcept>
#include <vector>

#include "util.cuh"
#include "methods.cuh"
#include "data.cuh"

std::vector<EMB_T> copyResults(Data data)
{
    std::vector<EMB_T> rst(data.config.numToScore * data.config.embDim);
    CHECK_CUDA(cudaMemcpy(rst.data(), data.d_rst, data.config.numToScore * data.config.embDim * sizeof(EMB_T), cudaMemcpyDeviceToHost));
    return rst;
}

float computeRMSE(std::vector<EMB_T> rstA, std::vector<EMB_T> rstB, int numToScore, int embDim)
{
    if (rstA.size() != rstB.size())
    {
        throw std::runtime_error("rstA.size() != rstB.size()");
    }

    double sum = 0;
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

int main()
{
    Config config;
    config.numDocs = 1000000;
    config.numToScore = 10000;
    config.embDim = 256;
    config.numBitsPerDim = 2;
    config.numCentroids = 10000;
    config.stdDev = 1.0f;
    config.debugMode = false;
    config.validate();

    Data data = genData(config);
    methodReference(data);
    std::vector<EMB_T> rstReference = copyResults(data);
    methodBaseline(data);
    std::vector<EMB_T> rstBaseline = copyResults(data);
    methodResQuant(data, false);
    std::vector<EMB_T> rstResQuant = copyResults(data);

    float rmseReference = computeRMSE(rstReference, rstReference, config.numToScore, config.embDim);
    float rmseBaseline = computeRMSE(rstBaseline, rstReference, config.numToScore, config.embDim);
    float rmseResQuant = computeRMSE(rstResQuant, rstReference, config.numToScore, config.embDim);
    std::cout << "rmseReference = " << rmseReference << ", rmseBaseline = " << rmseBaseline << ", rmseResQuant = " << rmseResQuant << std::endl;

    return 0;
}
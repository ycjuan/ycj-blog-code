#include <iostream>
#include <cassert>
#include <vector>
#include <sstream>

#include "data.cuh"

void testQuantize()
{
    int numBitsPerDim = 2;
    int numBitsPerInt = kBitsPerInt;
    float stdDev = 1.0f;
    {
        std::vector<float> residuals = {-2.1f, -2.0f, -1.5f, -1.0f, -0.5f, 0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 2.1f};
        std::vector<float> expectedResiduals = {-2.0f, -2.0f, -2.0f, -1.0f, -1.0f, 1.0f, 1.0f, 1.0f, 2.0f, 2.0f, 2.0f};

        for (int i = 0; i < residuals.size(); i++)
        {
            RQ_T quantRes = 0;
            float residual = residuals[i];
            for (int embIdx = 0; embIdx < numBitsPerInt / numBitsPerDim; embIdx++)
            {
                quantize(numBitsPerDim, numBitsPerInt, stdDev, residual, quantRes, embIdx);
                float recoveredResidual = dequantize(numBitsPerDim, numBitsPerInt, stdDev, quantRes, embIdx);
                if (recoveredResidual != expectedResiduals[i])
                {
                    std::ostringstream oss;
                    oss << "embIdx = " << embIdx << ", residual = " << residual << ", recovered residual = " << recoveredResidual << ", expected residual = " << expectedResiduals[i];
                    throw std::runtime_error(oss.str());
                }
            }
        }
    }

    std::cout << "testQuantize passed" << std::endl;
}

int main()
{
    testQuantize();

    return 0;
}
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
        std::vector<float> recoveredResiduals = {-2.0f, -2.0f, -2.0f, -1.0f, -1.0f, 1.0f, 1.0f, 1.0f, 2.0f, 2.0f, 2.0f};

        for (int i = 0; i < residuals.size(); i++)
        {
            RQ_T rq = 0;
            float residual = residuals[i];
            for (int embIdx = 0; embIdx < numBitsPerInt / numBitsPerDim; embIdx++)
            {
                quantize(numBitsPerDim, numBitsPerInt, stdDev, residual, rq, embIdx);
                float quantizedResidual = dequantize(numBitsPerDim, numBitsPerInt, stdDev, rq, embIdx);
                if (quantizedResidual != recoveredResiduals[i])
                {
                    std::ostringstream oss;
                    oss << "embIdx = " << embIdx << ", residual = " << residual << ", quantizedResidual = " << quantizedResidual << ", recoveredResidual = " << recoveredResiduals[i];
                    throw std::runtime_error(oss.str());
                }
            }
        }
    }
}

int main()
{
    float x = -0.7f;
    int y = (int)(std::floor(x));
    std::cout << "y = " << y << std::endl;
    testQuantize();

    return 0;
}
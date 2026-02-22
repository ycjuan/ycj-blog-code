#pragma once

#include <cstdint>
#include <cmath>
#include <cuda_runtime.h>

typedef uint64_t T_RQ;

constexpr int kBitsPerRqInt = 8 * sizeof(T_RQ);

inline __device__ __host__ size_t getRqDim(size_t embDim, size_t numBitsPerDim)
{
    return embDim * numBitsPerDim / kBitsPerRqInt;
}

inline __device__ __host__ int getRqIdx(int embIdx, int numBitsPerDim, int numBitsPerInt)
{
    int numEmbsPerInt = numBitsPerInt / numBitsPerDim;
    return embIdx / numEmbsPerInt;
}

inline __host__ void quantize(int numBitsPerDim, int numBitsPerInt, float stdDev, float residual, T_RQ& globalQuantRes, int embIdx)
{
    const int embsPerInt = numBitsPerInt / numBitsPerDim;
    const int shifts = (embIdx % embsPerInt) * numBitsPerDim;
    const int fullRange = (1 << numBitsPerDim);
    const int halfRange = (fullRange >> 1);

    int localQuantRes = residual > 0 ? (int)(std::ceil(residual / stdDev))
                                     : (int)(std::floor(residual / stdDev));
    localQuantRes = std::max(localQuantRes, -halfRange);
    localQuantRes = std::min(localQuantRes, halfRange);
    localQuantRes += halfRange;
    if (localQuantRes > halfRange)
    {
        localQuantRes--;
    }

    T_RQ mask = static_cast<T_RQ>(localQuantRes);
    mask <<= shifts;
    globalQuantRes |= mask;
}

inline __device__ __host__ float dequantize(int numBitsPerDim, int numBitsPerInt, float stdDev, T_RQ rq, int embIdx)
{
    const int embsPerInt = numBitsPerInt / numBitsPerDim;
    const int shifts = (embIdx % embsPerInt) * numBitsPerDim;
    const int fullRange = (1 << numBitsPerDim);
    const int halfRange = (fullRange >> 1);

    T_RQ mask = static_cast<T_RQ>(fullRange) - 1;
    mask <<= shifts;
    mask &= rq;
    mask >>= shifts;

    int localQuantRes = static_cast<int>(mask);
    if (localQuantRes >= halfRange)
    {
        localQuantRes++;
    }
    localQuantRes -= halfRange;

    return localQuantRes * stdDev;
}

#pragma once

#include <cuda_bf16.h>
#include <stdexcept>
#include <iostream>
#include <bitset>

#define EMB_T nv_bfloat16
#define RQ_T uint64_t

constexpr int kBitsPerInt = 8 * sizeof(RQ_T);

struct Config
{
    size_t numDocs = 0;
    size_t numToScore = 0;
    size_t embDim = 0;
    size_t numBitsPerDim = 0;
    size_t numCentroids = 0;
    float stdDev = 1.0f;
    bool debugMode = false;
    inline __device__ __host__ size_t getRqDim() const
    {
        return embDim * numBitsPerDim / kBitsPerInt;
    }

    void validate()
    {
        if (kBitsPerInt % numBitsPerDim != 0)
        {
            throw std::runtime_error("kBitsPerInt must be divisible by numBitsPerDim");
        }
    }
};

struct Data
{
    Config config;

    EMB_T* h_emb; // numDocs x embDim
    EMB_T* d_emb;
    EMB_T* h_centroidEmb; // numCentroids x embDim x 2 (the first half is the embedding, and the second half is the delta)
    EMB_T* d_centroidEmb;
    int* h_centroidIdx; // numDocs
    int* d_centroidIdx; // numDocs
    RQ_T* h_residual; // numDocs x embDim x numBitsPerDim / sizeof(RQ_T)
    RQ_T* d_residual;
    int* h_docIdxToScore; // numToScore
    int* d_docIdxToScore;
    EMB_T* d_rst; // numToScore x embDim
};

Data genData(Config config);

inline __device__ __host__ size_t getMemAddr(size_t i, size_t j, size_t M, size_t N)
{
    return (size_t)j * M + i;
}


inline __device__ __host__ int getRqIdx(int embIdx, int numBitsPerDim, int numBitsPerInt)
{
    return embIdx / (numBitsPerInt / numBitsPerDim);
}

template <typename T>
void printBits(T value, std::string name)
{
    std::cout << name << " = " << std::bitset<sizeof(T) * 8>(value) << std::endl;
}

inline void quantize(int numBitsPerDim, int numBitsPerInt, float stdDev, float residual, RQ_T& rq, int embIdx)
{
    std::cout << "========== quantize ==========" << std::endl;

    int embsPerInt = numBitsPerInt / numBitsPerDim;
    int bitOffset = (embIdx % embsPerInt) * numBitsPerDim;
    std::cout << "bitOffset = " << bitOffset << std::endl;
    RQ_T mask = (RQ_T(1) << numBitsPerDim) - 1;
    printBits(mask, "mask (0)");
    int quantizedResidual = residual > 0 ? (int)(std::ceil(residual / stdDev))
                                         : (int)(std::floor(residual / stdDev));
    std::cout << "quantizedResidual (0) = " << quantizedResidual << std::endl;
    quantizedResidual = std::max(quantizedResidual, -(1 << (numBitsPerDim - 1)));
    quantizedResidual = std::min(quantizedResidual, (1 << (numBitsPerDim - 1)));
    std::cout << "quantizedResidual (1) = " << quantizedResidual << std::endl;
    quantizedResidual += (1 << (numBitsPerDim - 1));
    std::cout << "quantizedResidual (2) = " << quantizedResidual << std::endl;
    if (quantizedResidual > (1 << (numBitsPerDim - 1)))
    {
        quantizedResidual--;
    }
    std::cout << "quantizedResidual (3) = " << quantizedResidual << std::endl;
    mask &= quantizedResidual;
    printBits(mask, "mask (1)");
    mask <<= bitOffset;
    printBits(mask, "mask (2)");
    printBits(rq, "rq (0)");
    rq |= mask;
    std::cout << "rq = " << rq << std::endl;
    printBits(rq, "rq (1)");
}

inline __host__ float retriveQuantRes(int numBitsPerDim, int numBitsPerInt, float stdDev, RQ_T rq, int embIdx)
{
    std::cout << "========== dequantize ==========" << std::endl;
    int embsPerInt = numBitsPerInt / numBitsPerDim;
    int bitOffset = (embIdx % embsPerInt) * numBitsPerDim;
    RQ_T mask = (RQ_T(1) << numBitsPerDim) - 1;
    std::cout << "embsPerInt = " << embsPerInt << std::endl;
    std::cout << "bitOffset = " << bitOffset << std::endl;
    printBits(mask, "mask (0)");

    mask <<= bitOffset;
    RQ_T quantizedResidual_ = rq & mask;
    quantizedResidual_ >>= bitOffset;
    int quantizedResidual = static_cast<int>(quantizedResidual_);
    std::cout << "quantizedResidual (0) = " << quantizedResidual << std::endl;
    if (quantizedResidual >= (1 << (numBitsPerDim - 1)))
    {
        quantizedResidual++;
    }
    std::cout << "quantizedResidual (1) = " << quantizedResidual << std::endl;
    quantizedResidual -= (1 << (numBitsPerDim - 1));
    std::cout << "quantizedResidual (2) = " << quantizedResidual << std::endl;
    return quantizedResidual * stdDev;
}
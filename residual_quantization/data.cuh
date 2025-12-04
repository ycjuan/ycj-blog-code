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
    EMB_T* h_centroidEmb; // numCentroids x embDim x 2 (the first half is the embedding, and the second half is the stdDev)
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
    return (size_t)i * N + j;
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

inline void quantize(int numBitsPerDim, int numBitsPerInt, float stdDev, float residual, RQ_T& globalQuantRes, int embIdx)
{
    // ----------------
    // We will use the following settings to demonstrate how this function works:
    //   numBitsPerDim = 2
    //   numBitsPerInt = 64 (we are using uint64_t)
    //   stdDev = 1.0f
    //   residual = {-2.1, -2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 2.1}
    //   embIdx = 1


    // ----------------
    // Initialize some constants
    //   embsPerInt: number of embeddings a given integer can represent. 
    //     For example, if we use uint64_t, and each embedding takes 2 bits, then embsPerInt = 64 / 2 = 32.
    //   shifts: this is used to tell where the 2 bits for a given embedding is located in the 64-bit integer.
    //     For example, if embIdx = 1, we want the 2 bits to be at 2nd and 3rd positions (0-indexed) in the 64-bit integer.
    //     So we need to do <<2, where 2 is obtained by (embIdx % embsPerInt) * numBitsPerDim = (1 % 32) * 2 = 2.
    //   fullRange: this is the full range of the 2 bits. 
    //     For example, if numBitsPerDim = 2, then fullRange = 4. (Meaning it can represent -2 * stdDev, -1 * stdDev, 1 * stdDev, 2 * stdDev)
    //   halfRange: half of the full range, so in our example, halfRange = 2. 
    //     This means how many stdDevs we can represent on either side of the mean.
    const int embsPerInt = numBitsPerInt / numBitsPerDim;
    const int shifts = (embIdx % embsPerInt) * numBitsPerDim;
    // In `quantize` we don't really need fullRange, but we will need it in `dequantize`.
    // For code consistency, we will still calculate it here and use it to infer halfRange.
    const int fullRange = (1 << numBitsPerDim);
    const int halfRange = (fullRange >> 1);

    // ----------------
    // The following code will perform the following mapping:
    //   * [-inf, -1.0) => 0
    //   * [-1.0,    0) => 1
    //   * [   0,  1.0] => 2
    //   * ( 1.0,  inf] => 3
    // It's best to run through this example to understand how it works:
    //   * -2.1 => floor(-2.1 / 1.0) = -3 => max(-3, -2) = -2 => min(-2, 2) = -2 => -2 + 2 = 0
    //   * -2.0 => floor(-2.0 / 1.0) = -2 => max(-2, -2) = -2 => min(-2, 2) = -2 => -2 + 2 = 0
    //   * -1.5 => floor(-1.5 / 1.0) = -2 => max(-2, -2) = -2 => min(-2, 2) = -2 => -2 + 2 = 0
    //   * -1.0 => floor(-1.0 / 1.0) = -1 => max(-1, -2) = -1 => min(-1, 2) = -1 => -1 + 2 = 1
    //   * -0.5 => floor(-0.5 / 1.0) = -1 => max(-1, -2) = -1 => min(-1, 2) = -1 => -1 + 2 = 1
    //   *  0.0 => floor( 0.0 / 1.0) =  0 => max( 0, -2) =  0 => min( 0, 2) =  0 =>  0 + 2 = 2
    //   *  0.5 =>  ceil( 0.5 / 1.0) =  1 => max( 1, -2) =  1 => min( 1, 2) =  1 =>  1 + 2 = 3 => 3 - 1 = 2
    //   *  1.0 =>  ceil( 1.0 / 1.0) =  1 => max( 1, -2) =  1 => min( 1, 2) =  1 =>  1 + 2 = 3 => 3 - 1 = 2
    //   *  1.5 =>  ceil( 1.5 / 1.0) =  2 => max( 2, -2) =  2 => min( 2, 2) =  2 =>  2 + 2 = 4 => 4 - 1 = 3
    //   *  2.0 =>  ceil( 2.0 / 1.0) =  2 => max( 2, -2) =  2 => min( 2, 2) =  2 =>  2 + 2 = 4 => 4 - 1 = 3
    //   *  2.1 =>  ceil( 2.1 / 1.0) =  3 => max( 3, -2) =  3 => min( 3, 2) =  2 =>  2 + 2 = 4 => 4 - 1 = 3
    // VERY IMPORTANT: We cannot use unsigned integer here because we need to handle the case when residual is negative.
    int localQuantRes = residual > 0 ? (int)(std::ceil(residual / stdDev))
                                         : (int)(std::floor(residual / stdDev));
    localQuantRes = std::max(localQuantRes, -halfRange);
    localQuantRes = std::min(localQuantRes, halfRange);
    localQuantRes += halfRange;
    if (localQuantRes > halfRange)
    {
        // Without this subtraction, if you run the above example carefully, you will find a problem that:
        //   * only "EXACT ZERO" got mapped to 2
        //   * residual between (0.0f, 1.0f] got mapped to 3
        //   * residual between (1.0f, inf] got mapped to 4
        // This is undisirable because:
        //   1. It is wasting to use a bit to record "EXACT ZERO" because in real world, the probability of getting "EXACT ZERO" is extremely low.
        //   2. We cannot use 2 bits to represent "4"
        // Therefore, we will subtract 1 in "right side of the mean" to make it:
        //   * [0.0, 1.0] => 2
        //   * (1.0, inf] => 3
        localQuantRes--;
    }

    // ----------------
    // Use a mask to encode the quantized residual into the 64-bit integer
    // At this point, quantizedResidual = 0, 1, 2, 3 (which is 0b00, 0b01, 0b10, 0b11), we first cast it to a 64-bit integer.
    RQ_T mask = static_cast<RQ_T>(localQuantRes);
    // Then, we shift the mask to the desired position.
    mask <<= shifts;
    // Finally, we perform a bitwise OR with the existing residual so that the quantized residual is encoded into the 64-bit integer.
    // !!!!!VERY IMPORTANT!!!!! rq is NOT OVERWRITTEN-ABLE. Let's say you first encode 0b01, and then encode 0b10, the result will be 0b11, NOT 0b10
    globalQuantRes |= mask;
}

inline __device__ __host__ float dequantize(int numBitsPerDim, int numBitsPerInt, float stdDev, RQ_T rq, int embIdx)
{
    // We will use the same example as in `quantize` to demonstrate how this function works.
    // Before reading this function, please make sure you understand the comments in `quantize` first.

    // ----------------
    // Initialize some constants (Please refer to the comments in `quantize` for more details)
    const int embsPerInt = numBitsPerInt / numBitsPerDim;
    const int shifts = (embIdx % embsPerInt) * numBitsPerDim;
    const int fullRange = (1 << numBitsPerDim);
    const int halfRange = (fullRange >> 1);

    // ----------------
    // Extract the quantized residual from the 64-bit integer
    // For example, when numBitsPerDim = 2, we have fullRange = 4, so mask = 4 - 1 = 3 = 0b11
    RQ_T mask = static_cast<RQ_T>(fullRange) - 1;
    // We left shift the mask to position that stores the quantized residual of the embedding at `embIdx`
    mask <<= shifts;
    // We perform a bitwise AND with the 64-bit integer to extract the quantized residual of the embedding at `embIdx`
    mask &= rq;
    // We right shift the mask to the original position. When numBitsPerDim = 2, the value of mask will be one of {0b00 (0), 0b01 (1), 0b10 (2), 0b11 (3)}
    mask >>= shifts;
    // We cast the mask to a signed integer because it may be negative after running the code below
    int localQuantRes = static_cast<int>(mask);
    // Revert the `localQuantRes--` trick in `quantize`. Note that in `quantize`, we need `>`, but here we need `>=`.
    if (localQuantRes >= halfRange)
    {
        localQuantRes++;
    }
    // Revert the `localQuantRes += halfRange` trick in `quantize`. Now the localQuantRes would be one of {-2, -1, 1, 2}
    localQuantRes -= halfRange;
    // Finally, we multiply the localQuantRes by the stdDev to get the recovered residual
    return localQuantRes * stdDev;
}
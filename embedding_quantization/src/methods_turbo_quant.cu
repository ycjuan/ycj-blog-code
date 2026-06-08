#include "data.cuh"
#include "methods_turbo_quant.cuh"
#include "util.cuh"

// One block per scored document, 1024 threads per block.
// Each thread handles kElemsPerThread=4 elements (assumes embDim=4096).
//
// WHT butterfly is split into three tiers to minimize shared memory pressure:
//   - Strides 1, 2        (< kElemsPerThread):     intra-thread, pure register ops
//   - Strides 4 .. 64     (intra-warp):             __shfl_xor_sync, no shared memory
//   - Strides 128 .. 2048 (inter-warp):             single shared memory buffer (16 KB)
//
// Using a single buffer (vs ping-pong) halves shared memory from 32 KB to 16 KB,
// doubling occupancy from 4 to 8 blocks per SM.
__global__ void turboQuantKernel(Data data, RQ_T* p_turboRes)
{
    constexpr int kElemsPerThread = 4;

    int toScoreIdx = blockIdx.x;
    if (toScoreIdx >= data.config.numToScore)
        return;

    int docIdx = data.d_docIdxToScore[toScoreIdx];
    int embDim = data.config.embDim;

    // Step 1: Lloyd-Max dequantize into registers (rotated space)
    float vals[kElemsPerThread];
    for (int k = 0; k < kElemsPerThread; k++)
    {
        int    embIdx    = threadIdx.x * kElemsPerThread + k;
        int    rqIdx     = getRqIdx(embIdx, data.config.numBitsPerDim, kBitsPerInt);
        size_t rqMemAddr = getMemAddr(docIdx, rqIdx, data.config.numDocs, data.config.getRqDim());
        vals[k]          = lloydMaxDequantize(data.config.numBitsPerDim,
                                     kBitsPerInt,
                                     data.config.stdDev,
                                     p_turboRes[rqMemAddr],
                                     embIdx);
    }

    // Step 2: WHT butterfly — inverse RHT = (1/sqrt(d)) * D * WHT(x_rot)

    // Tier 1: strides 1, 2 — pairs lie within the same thread's 4 elements, no sync needed
    {
        float a;
        a       = vals[0];
        vals[0] = a + vals[1];
        vals[1] = a - vals[1];
        a       = vals[2];
        vals[2] = a + vals[3];
        vals[3] = a - vals[3];
        a       = vals[0];
        vals[0] = a + vals[2];
        vals[2] = a - vals[2];
        a       = vals[1];
        vals[1] = a + vals[3];
        vals[3] = a - vals[3];
    }

    // Tier 2: strides 4..64 — pairs within the same warp, use warp shuffles
    for (int stride = kElemsPerThread; stride < kElemsPerThread * 32; stride <<= 1)
    {
        int laneMask = stride / kElemsPerThread;
        for (int k = 0; k < kElemsPerThread; k++)
        {
            int   globalIdx = threadIdx.x * kElemsPerThread + k;
            float partner   = __shfl_xor_sync(0xffffffff, vals[k], laneMask);
            vals[k]         = (globalIdx & stride) ? (partner - vals[k]) : (vals[k] + partner);
        }
    }

    // Tier 3: strides 128..2048 — inter-warp, single shared memory buffer (16 KB)
    extern __shared__ float shm[]; // embDim floats

    for (int k = 0; k < kElemsPerThread; k++)
        shm[threadIdx.x * kElemsPerThread + k] = vals[k];
    __syncthreads();

    for (int stride = kElemsPerThread * 32; stride < embDim; stride <<= 1)
    {
        for (int k = 0; k < kElemsPerThread; k++)
        {
            int   globalIdx  = threadIdx.x * kElemsPerThread + k;
            float localVal   = shm[globalIdx];
            float partnerVal = shm[globalIdx ^ stride];
            vals[k]          = (globalIdx & stride) ? (partnerVal - localVal) : (localVal + partnerVal);
        }
        __syncthreads();
        // Skip the write-back on the last stride — vals[] already holds the final result
        if (stride < embDim / 2)
        {
            for (int k = 0; k < kElemsPerThread; k++)
                shm[threadIdx.x * kElemsPerThread + k] = vals[k];
            __syncthreads();
        }
    }

    // Step 3: sign flip, normalize, add centroid, store
    int   centroidIdx = data.d_centroidIdx[docIdx];
    float scale       = 1.0f / sqrtf((float)embDim);
    for (int k = 0; k < kElemsPerThread; k++)
    {
        int    embIdx          = threadIdx.x * kElemsPerThread + k;
        float  residual        = vals[k] * scale * data.d_rhtSigns[embIdx];
        size_t centroidAddr    = getMemAddr(centroidIdx, embIdx, data.config.numCentroids, data.config.embDim);
        float  centroid        = static_cast<float>(data.d_centroidVal[centroidAddr]);
        size_t dstMemAddr      = getMemAddr(toScoreIdx, embIdx, data.config.numToScore, data.config.embDim);
        data.d_rst[dstMemAddr] = static_cast<EMB_T>(centroid + residual);
    }
}

void methodTurboQuant(Data data, bool copyTurboResFromHost)
{
    RQ_T*  p_turboRes = copyTurboResFromHost ? data.h_turboRes : data.d_turboRes;
    size_t shmBytes   = data.config.embDim * sizeof(float); // single buffer, 16 KB for embDim=4096
    turboQuantKernel<<<data.config.numToScore, 1024, shmBytes>>>(data, p_turboRes);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());
}

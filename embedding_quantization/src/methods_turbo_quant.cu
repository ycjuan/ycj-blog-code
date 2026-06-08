#include "data.cuh"
#include "methods_turbo_quant.cuh"
#include "util.cuh"

// One block per scored document, 1024 threads per block.
// elemsPerThread = embDim / 1024 (e.g. 1 for embDim=1024, 4 for embDim=4096).
//
// WHT butterfly is split into three tiers to minimize shared memory pressure:
//   - Strides < elemsPerThread:                intra-thread, pure register ops
//   - Strides elemsPerThread .. 16*elemsPerThread (intra-warp): __shfl_xor_sync
//   - Strides 32*elemsPerThread .. embDim/2    (inter-warp):   single shared memory buffer
//
// Using a single buffer (vs ping-pong) halves shared memory, doubling occupancy.

constexpr int kMaxElemsPerThread = 4; // upper bound; actual value is embDim / 1024

__global__ void turboQuantKernel(Data data, RQ_T* p_turboRes)
{
    int toScoreIdx = blockIdx.x;
    if (toScoreIdx >= data.config.numToScore)
        return;

    int   docIdx         = data.d_docIdxToScore[toScoreIdx];
    int   centroidIdx    = data.d_centroidIdx[docIdx];
    int   embDim         = data.config.embDim;
    int   elemsPerThread = embDim / blockDim.x;
    float effStd         = data.d_centroidEffStd[centroidIdx];

    // Step 1: Lloyd-Max dequantize into registers (rotated space)
    float vals[kMaxElemsPerThread];
    for (int k = 0; k < elemsPerThread; k++)
    {
        int    embIdx    = threadIdx.x * elemsPerThread + k;
        int    rqIdx     = getRqIdx(embIdx, data.config.numBitsPerDim, kBitsPerInt);
        size_t rqMemAddr = getMemAddr(docIdx, rqIdx, data.config.numDocs, data.config.getRqDim());
        vals[k] = lloydMaxDequantize(data.config.numBitsPerDim, kBitsPerInt, effStd, p_turboRes[rqMemAddr], embIdx);
    }

    // Step 2: WHT butterfly — inverse RHT = (1/sqrt(d)) * D * WHT(x_rot)

    // Tier 1: strides 1..(elemsPerThread-1) — pairs within this thread's elements
    for (int stride = 1; stride < elemsPerThread; stride <<= 1)
    {
        float tmp[kMaxElemsPerThread];
        for (int k = 0; k < elemsPerThread; k++)
            tmp[k] = vals[k];
        for (int k = 0; k < elemsPerThread; k++)
            vals[k] = (k & stride) ? (tmp[k ^ stride] - tmp[k]) : (tmp[k] + tmp[k ^ stride]);
    }

    // Tier 2: strides elemsPerThread..16*elemsPerThread — intra-warp, use warp shuffles
    for (int stride = elemsPerThread; stride < elemsPerThread * 32; stride <<= 1)
    {
        int laneMask = stride / elemsPerThread;
        for (int k = 0; k < elemsPerThread; k++)
        {
            int   globalIdx = threadIdx.x * elemsPerThread + k;
            float partner   = __shfl_xor_sync(0xffffffff, vals[k], laneMask);
            vals[k]         = (globalIdx & stride) ? (partner - vals[k]) : (vals[k] + partner);
        }
    }

    // Tier 3: strides 32*elemsPerThread..embDim/2 — inter-warp, single shared memory buffer
    extern __shared__ float shm[]; // embDim floats

    for (int k = 0; k < elemsPerThread; k++)
        shm[threadIdx.x * elemsPerThread + k] = vals[k];
    __syncthreads();

    for (int stride = elemsPerThread * 32; stride < embDim; stride <<= 1)
    {
        for (int k = 0; k < elemsPerThread; k++)
        {
            int   globalIdx  = threadIdx.x * elemsPerThread + k;
            float localVal   = shm[globalIdx];
            float partnerVal = shm[globalIdx ^ stride];
            vals[k]          = (globalIdx & stride) ? (partnerVal - localVal) : (localVal + partnerVal);
        }
        __syncthreads();
        if (stride < embDim / 2)
        {
            for (int k = 0; k < elemsPerThread; k++)
                shm[threadIdx.x * elemsPerThread + k] = vals[k];
            __syncthreads();
        }
    }

    // Step 3: sign flip, normalize, add centroid, store
    float scale = 1.0f / sqrtf((float)embDim);
    for (int k = 0; k < elemsPerThread; k++)
    {
        int    embIdx          = threadIdx.x * elemsPerThread + k;
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
    size_t shmBytes   = data.config.embDim * sizeof(float);
    turboQuantKernel<<<data.config.numToScore, 1024, shmBytes>>>(data, p_turboRes);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());
}

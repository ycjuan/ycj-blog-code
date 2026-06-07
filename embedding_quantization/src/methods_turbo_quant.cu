#include "data.cuh"
#include "methods_turbo_quant.cuh"
#include "util.cuh"

// One block per scored document, 1024 threads per block.
// Each thread handles (embDim / 1024) elements cooperatively.
// Shared memory layout: two ping-pong buffers of embDim floats for the WHT butterfly.
__global__ void turboQuantKernel(Data data, RQ_T* p_turboRes)
{
    int toScoreIdx = blockIdx.x;
    if (toScoreIdx >= data.config.numToScore)
        return;

    int docIdx         = data.d_docIdxToScore[toScoreIdx];
    int embDim         = data.config.embDim;
    int elemsPerThread = embDim / blockDim.x;

    // Two ping-pong buffers for the in-place WHT butterfly
    extern __shared__ float shm[]; // 2 * embDim floats
    float*                  cur = shm;
    float*                  nxt = shm + embDim;

    // Step 1: Lloyd-Max dequantize into the rotated space
    for (int k = 0; k < elemsPerThread; k++)
    {
        int    embIdx    = threadIdx.x * elemsPerThread + k;
        int    rqIdx     = getRqIdx(embIdx, data.config.numBitsPerDim, kBitsPerInt);
        size_t rqMemAddr = getMemAddr(docIdx, rqIdx, data.config.numDocs, data.config.getRqDim());
        cur[embIdx]      = lloydMaxDequantize(data.config.numBitsPerDim,
                                         kBitsPerInt,
                                         data.config.stdDev,
                                         p_turboRes[rqMemAddr],
                                         embIdx);
    }
    __syncthreads();

    // Step 2: Inverse RHT = (1/sqrt(d)) * D * WHT
    // The forward RHT was: x_rot = (1/sqrt(d)) * WHT(D * x)
    // The inverse is:      x     = (1/sqrt(d)) * D * WHT(x_rot)
    // where D = diag(h_rhtSigns). Both forward and inverse have identical cost.

    // WHT butterfly (ping-pong to avoid read-write conflicts)
    for (int stride = 1; stride < embDim; stride <<= 1)
    {
        for (int k = 0; k < elemsPerThread; k++)
        {
            int   embIdx = threadIdx.x * elemsPerThread + k;
            float a      = cur[embIdx];
            float b      = cur[embIdx ^ stride];
            nxt[embIdx]  = (embIdx & stride) ? (b - a) : (a + b);
        }
        __syncthreads();
        float* tmp = cur;
        cur        = nxt;
        nxt        = tmp;
    }

    // Apply sign flip and normalize, then add centroid to recover the embedding
    int   centroidIdx = data.d_centroidIdx[docIdx];
    float scale       = 1.0f / sqrtf((float)embDim);
    for (int k = 0; k < elemsPerThread; k++)
    {
        int    embIdx          = threadIdx.x * elemsPerThread + k;
        float  residual        = cur[embIdx] * scale * data.d_rhtSigns[embIdx];
        size_t centroidAddr    = getMemAddr(centroidIdx, embIdx * 2, data.config.numCentroids, data.config.embDim * 2);
        float  centroid        = static_cast<float>(data.d_centroidEmb[centroidAddr]);
        size_t dstMemAddr      = getMemAddr(toScoreIdx, embIdx, data.config.numToScore, data.config.embDim);
        data.d_rst[dstMemAddr] = static_cast<EMB_T>(centroid + residual);
    }
}

void methodTurboQuant(Data data, bool copyTurboResFromHost)
{
    RQ_T*  p_turboRes = copyTurboResFromHost ? data.h_turboRes : data.d_turboRes;
    size_t shmBytes   = 2 * data.config.embDim * sizeof(float);
    turboQuantKernel<<<data.config.numToScore, 1024, shmBytes>>>(data, p_turboRes);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());
}

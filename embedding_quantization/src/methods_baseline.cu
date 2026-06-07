#include "data.cuh"
#include "methods_baseline.cuh"
#include "util.cuh"

constexpr int kBlockSize = 1024;

__global__ void baselineKernel(Data data, EMB_T* p_emb)
{
    size_t tidx       = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    int    toScoreIdx = tidx / data.config.embDim;
    int    embIdx     = tidx % data.config.embDim;
    int    docIdx     = data.d_docIdxToScore[toScoreIdx];
    if (toScoreIdx < data.config.numToScore)
    {
        size_t srcMemAddr      = getMemAddr(docIdx, embIdx, data.config.numDocs, data.config.embDim);
        size_t dstMemAddr      = getMemAddr(toScoreIdx, embIdx, data.config.numToScore, data.config.embDim);
        data.d_rst[dstMemAddr] = p_emb[srcMemAddr];
    }
}

void methodBaseline(Data data, bool copyEmbFromHost)
{
    EMB_T* p_emb    = (copyEmbFromHost) ? data.h_emb : data.d_emb;
    size_t gridSize = (data.config.numToScore * data.config.embDim + kBlockSize - 1) / kBlockSize;
    baselineKernel<<<gridSize, kBlockSize>>>(data, p_emb);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());
}

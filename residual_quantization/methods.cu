#include <vector>
#include "data.cuh"
#include "util.cuh"

void methodReference(Data data)
{
    std::vector<EMB_T> v_rst(data.config.numToScore * data.config.embDim);
    for (int toBeScoredIdx = 0; toBeScoredIdx < data.config.numToScore; toBeScoredIdx++)
    {
        auto docIdx = data.h_docIdxToScore[toBeScoredIdx];
        for (int embIdx = 0; embIdx < data.config.embDim; embIdx++)
        {
            auto srcMemAddr = getMemAddr(docIdx, embIdx, data.config.numDocs, data.config.embDim);
            auto dstMemAddr = getMemAddr(toBeScoredIdx, embIdx, data.config.numToScore, data.config.embDim);
            v_rst[dstMemAddr] = data.h_emb[srcMemAddr];
        }
    }

    CHECK_CUDA(cudaMemcpy(data.d_rst, v_rst.data(), data.config.numToScore * data.config.embDim * sizeof(EMB_T), cudaMemcpyHostToDevice));
}

__global__ void baselineKernel(Data data, EMB_T* p_emb)
{
    int tidx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    int toBeScoredIdx = tidx / data.config.embDim;
    int embIdx = tidx % data.config.embDim;
    int docIdx = data.h_docIdxToScore[toBeScoredIdx];
    if (toBeScoredIdx < data.config.numToScore)
    {
        size_t srcMemAddr = getMemAddr(docIdx, embIdx, data.config.numDocs, data.config.embDim);
        size_t dstMemAddr = getMemAddr(toBeScoredIdx, embIdx, data.config.numToScore, data.config.embDim);
        data.d_rst[dstMemAddr] = p_emb[srcMemAddr];
    }
}

void methodBaseline(Data data, bool copyEmbFromHost)
{
    EMB_T* p_emb = (copyEmbFromHost) ? data.h_emb : data.d_emb;
    baselineKernel<<<data.config.numToScore * data.config.embDim / 1024, 1024>>>(data, p_emb);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());
}

__global__ void resQuantKernel(Data data, RQ_T* p_residual)
{
    int tidx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    int toBeScoredIdx = tidx / data.config.embDim;
    int embIdx = tidx % data.config.embDim;
    int rqIdx = getRqIdx(embIdx, data.config.numBitsPerDim, kBitsPerInt);
    if (toBeScoredIdx < data.config.numToScore)
    {
        int docIdx = data.h_docIdxToScore[toBeScoredIdx];
        int centroidIdx = data.h_centroidIdx[docIdx];

        size_t centroidMemAddr = getMemAddr(centroidIdx, embIdx, data.config.numCentroids, data.config.embDim * 2);
        size_t stdDevMemAddr = getMemAddr(centroidIdx, embIdx + data.config.embDim, data.config.numCentroids, data.config.embDim * 2);
        size_t rqMemAddr = getMemAddr(docIdx, rqIdx, data.config.numDocs, data.config.getRqDim());

        EMB_T centroid = data.d_centroidEmb[centroidMemAddr];
        EMB_T stdDev = data.d_centroidEmb[stdDevMemAddr];
        RQ_T rq = p_residual[rqMemAddr];

        EMB_T residual = dequantize(data.config.numBitsPerDim, kBitsPerInt, data.config.stdDev, rq, embIdx);

        EMB_T rst = centroid + residual;

        size_t rstMemAddr = getMemAddr(toBeScoredIdx, embIdx, data.config.numToScore, data.config.embDim);
        data.d_rst[rstMemAddr] = rst;
    }
}

void methodResQuant(Data data, bool copyResidualFromHost)
{
    RQ_T* p_residual = (copyResidualFromHost) ? data.h_residual : data.d_residual;
    resQuantKernel<<<data.config.numToScore * data.config.embDim / 1024, 1024>>>(data, p_residual);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());
}
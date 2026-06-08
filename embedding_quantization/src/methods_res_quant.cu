#include "data.cuh"
#include "methods_res_quant.cuh"
#include "util.cuh"

constexpr int kBlockSize = 1024;

__global__ void resQuantKernel(Data data, RQ_T* p_quantRes)
{
    // tidx ranges from 0 to numToScore * embDim - 1, so we use `/` and `%` to get the toScoreIdx and embIdx
    // respectively
    size_t tidx       = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    int    toScoreIdx = tidx / data.config.embDim;
    int    embIdx     = tidx % data.config.embDim;
    // get the index of the quantized residual
    int quantResIdx = getRqIdx(embIdx, data.config.numBitsPerDim, kBitsPerInt);
    if (toScoreIdx < data.config.numToScore)
    {
        // get the document index and centroid index
        int docIdx      = data.d_docIdxToScore[toScoreIdx];
        int centroidIdx = data.d_centroidIdx[docIdx];

        // get the memory address of the centroid and quantized residual
        size_t centroidMemAddr = getMemAddr(centroidIdx, embIdx, data.config.numCentroids, data.config.embDim);
        size_t quantResMemAddr = getMemAddr(docIdx, quantResIdx, data.config.numDocs, data.config.getRqDim());

        // get the centroid, per-dim stdDev, and quantized residual
        EMB_T centroid = data.d_centroidVal[centroidMemAddr];
        float stdDev   = data.d_centroidStd[centroidMemAddr];
        RQ_T  quantRes = p_quantRes[quantResMemAddr];

        // perform the dequantization
        EMB_T residual = dequantize(data.config.numBitsPerDim, kBitsPerInt, stdDev, quantRes, embIdx);

        // recover the embedding value by adding the centroid and the residual
        EMB_T rst = centroid + residual;

        // get the memory address of the result
        size_t rstMemAddr = getMemAddr(toScoreIdx, embIdx, data.config.numToScore, data.config.embDim);

        // store the result
        data.d_rst[rstMemAddr] = rst;
    }
}

void methodResQuant(Data data, bool copyResidualFromHost)
{
    RQ_T*  p_residual = (copyResidualFromHost) ? data.h_residual : data.d_residual;
    size_t gridSize   = (data.config.numToScore * data.config.embDim + kBlockSize - 1) / kBlockSize;
    resQuantKernel<<<gridSize, kBlockSize>>>(data, p_residual);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());
}

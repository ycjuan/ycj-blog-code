#include <vector>
#include "data.cuh"
#include "util.cuh"
#include "methods.cuh"

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
    int docIdx = data.d_docIdxToScore[toBeScoredIdx];
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
    size_t gridSize = (data.config.numToScore * data.config.embDim + 1023) / 1024;
    baselineKernel<<<gridSize, 1024>>>(data, p_emb);
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
        int docIdx = data.d_docIdxToScore[toBeScoredIdx];
        int centroidIdx = data.d_centroidIdx[docIdx];

        size_t centroidMemAddr = getMemAddr(centroidIdx, embIdx * 2, data.config.numCentroids, data.config.embDim * 2);
        size_t rqMemAddr = getMemAddr(docIdx, rqIdx, data.config.numDocs, data.config.getRqDim());

        EMB_T centroid = data.d_centroidEmb[centroidMemAddr];
        EMB_T stdDev = data.d_centroidEmb[centroidMemAddr + 1];
        RQ_T rq = p_residual[rqMemAddr];

        EMB_T residual = dequantize(data.config.numBitsPerDim, kBitsPerInt, stdDev, rq, embIdx);

        EMB_T rst = centroid + residual;

        size_t rstMemAddr = getMemAddr(toBeScoredIdx, embIdx, data.config.numToScore, data.config.embDim);
        data.d_rst[rstMemAddr] = rst;
    }
}

void methodResQuant(Data data, bool copyResidualFromHost)
{
    RQ_T* p_residual = (copyResidualFromHost) ? data.h_residual : data.d_residual;
    size_t gridSize = (data.config.numToScore * data.config.embDim + 1023) / 1024;
    resQuantKernel<<<gridSize, 1024>>>(data, p_residual);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());
}

void runMethod(Data data, Method method)
{
    switch (method)
    {
        case Method::REFERENCE:
            methodReference(data);
            break;
        case Method::BASELINE_H2D:
            methodBaseline(data, true);
            break;
        case Method::BASELINE_D2D:
            methodBaseline(data, false);
            break;
        case Method::RES_QUANT_H2D:
            methodResQuant(data, true);
            break;
        case Method::RES_QUANT_D2D:
            methodResQuant(data, false);
            break;
        default:
            throw std::runtime_error("Invalid method");
    }
}
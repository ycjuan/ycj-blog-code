#include <vector>
#include "data.cuh"
#include "util.cuh"
#include "methods.cuh"

constexpr int kBlockSize = 1024;

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
    size_t tidx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    int toScoreIdx = tidx / data.config.embDim;
    int embIdx = tidx % data.config.embDim;
    int docIdx = data.d_docIdxToScore[toScoreIdx];
    if (toScoreIdx < data.config.numToScore)
    {
        size_t srcMemAddr = getMemAddr(docIdx, embIdx, data.config.numDocs, data.config.embDim);
        size_t dstMemAddr = getMemAddr(toScoreIdx, embIdx, data.config.numToScore, data.config.embDim);
        data.d_rst[dstMemAddr] = p_emb[srcMemAddr];
    }
}

void methodBaseline(Data data, bool copyEmbFromHost)
{
    EMB_T* p_emb = (copyEmbFromHost) ? data.h_emb : data.d_emb;
    size_t gridSize = (data.config.numToScore * data.config.embDim + kBlockSize - 1) / kBlockSize;
    baselineKernel<<<gridSize, kBlockSize>>>(data, p_emb);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());
}

__global__ void resQuantKernel(Data data, RQ_T* p_quantRes)
{
    // tidx ranges from 0 to numToScore * embDim - 1, so we use `/` and `%` to get the toScoreIdx and embIdx respectively
    size_t tidx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    int toScoreIdx = tidx / data.config.embDim;
    int embIdx = tidx % data.config.embDim;
    // get the index of the quantized residual
    int quantResIdx = getRqIdx(embIdx, data.config.numBitsPerDim, kBitsPerInt);
    if (toScoreIdx < data.config.numToScore)
    {
        // get the document index and centroid index
        int docIdx = data.d_docIdxToScore[toScoreIdx];
        int centroidIdx = data.d_centroidIdx[docIdx];

        // get the memory address of the centroid and quantized residual
        size_t centroidMemAddr = getMemAddr(centroidIdx, embIdx * 2, data.config.numCentroids, data.config.embDim * 2);
        size_t quantResMemAddr = getMemAddr(docIdx, quantResIdx, data.config.numDocs, data.config.getRqDim());

        // get the centroid, stdDev, and quantized residual
        EMB_T centroid = data.d_centroidEmb[centroidMemAddr];
        EMB_T stdDev = data.d_centroidEmb[centroidMemAddr + 1];
        RQ_T quantRes = p_quantRes[quantResMemAddr];

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
    RQ_T* p_residual = (copyResidualFromHost) ? data.h_residual : data.d_residual;
    size_t gridSize = (data.config.numToScore * data.config.embDim + kBlockSize - 1) / kBlockSize;
    resQuantKernel<<<gridSize, kBlockSize>>>(data, p_residual);
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
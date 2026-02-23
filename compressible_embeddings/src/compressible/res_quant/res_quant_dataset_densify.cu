#include <algorithm>

#include "common/typedef.hpp"
#include "compressible/res_quant/res_quant_dataset.hpp"
#include "utils/util.hpp"

struct DensifyFromResQuantKernelParams
{
    // Centroid data
    const T_EMB* d_centroidEmb;
    int numCentroids;

    // RQ config
    int numBitsPerDim;
    int rqDim;

    // Source
    const int* d_srcCentroidIdx;
    const T_RQ* h_srcResidual;
    int srcEmbDim;
    T_DOC_IDX srcNumDocs;

    // Destination
    T_EMB* d_dstEmbData;
    int dstEmbOffset;
    T_DOC_IDX dstNumDocs;
    int dstEmbDim;

    // Copy tasks
    int embDimToCopyBeginIncl;
    int embDimToCopyEndExcl;
    CopyTask* d_copyTasks;
    int numCopyTasks;
};

__global__ void densifyFromResQuantKernel(DensifyFromResQuantKernelParams params)
{
    size_t tidx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    int compressibleEmbDim = params.embDimToCopyEndExcl - params.embDimToCopyBeginIncl;
    int copyIdx = tidx / compressibleEmbDim;
    int localEmbIdx = tidx % compressibleEmbDim;

    if (copyIdx < params.numCopyTasks)
    {
        CopyTask task = params.d_copyTasks[copyIdx];
        T_DOC_IDX docIdx = task.srcDocIdx;
        int toScoreIdx = task.dstDocIdx;

        int globalEmbIdx = localEmbIdx + params.embDimToCopyBeginIncl;
        int centroidIdx = params.d_srcCentroidIdx[docIdx];

        // Get centroid value and stdDev
        size_t centroidMemAddr
            = getMemAddrRowMajor(centroidIdx, globalEmbIdx * 2, params.numCentroids, params.srcEmbDim * 2);
        T_EMB centroid = params.d_centroidEmb[centroidMemAddr];
        float stdDev = static_cast<float>(params.d_centroidEmb[centroidMemAddr + 1]);

        // Get quantized residual and dequantize
        int rqIdx = getRqIdx(globalEmbIdx, params.numBitsPerDim, kBitsPerRqInt);
        size_t rqMemAddr = getMemAddrRowMajor(docIdx, rqIdx, params.srcNumDocs, params.rqDim);
        T_RQ quantRes = params.h_srcResidual[rqMemAddr];
        float residual = dequantize(params.numBitsPerDim, kBitsPerRqInt, stdDev, quantRes, globalEmbIdx);

        // Reconstruct: centroid + residual
        T_EMB rst = static_cast<T_EMB>(static_cast<float>(centroid) + residual);

        // Write to working index
        size_t dstMemAddr = getMemAddrRowMajor(toScoreIdx,
                                               localEmbIdx + params.dstEmbOffset,
                                               params.dstNumDocs,
                                               params.dstEmbDim);
        params.d_dstEmbData[dstMemAddr] = rst;
    }
}

void ResQuantDataset::densifyCompressible(const DensificationTask& densificationTask) const
{
    int globalEmbDim = m_rqDim * kBitsPerRqInt / m_numBitsPerDim;

    for (const auto& compressibleConfig : m_compressiblePartitionConfigs)
    {
        // Compute the overlap between this compressible partition and the requested range.
        int embDimBeginReal
            = std::max(densificationTask.globalEmbIdxBeginIncl, (int)compressibleConfig.getEmbDimBeginIncl());
        int embDimEndReal = std::min(densificationTask.globalEmbIdxEndExcl, (int)compressibleConfig.getEmbDimEndExcl());

        if (embDimBeginReal >= embDimEndReal)
        {
            continue; // No overlap
        }

        int compressibleEmbDim = embDimEndReal - embDimBeginReal;

        DensifyFromResQuantKernelParams params;
        params.d_centroidEmb = m_d_centroidEmb.data();
        params.numCentroids = m_numCentroids;
        params.srcEmbDim = globalEmbDim;
        params.d_srcCentroidIdx = m_d_centroidIdx.data();
        params.h_srcResidual = m_h_residual.data();
        params.srcNumDocs = m_d_centroidIdx.getArraySize();
        params.rqDim = m_rqDim;
        params.numBitsPerDim = m_numBitsPerDim;
        params.dstNumDocs = densificationTask.desiredDocIdxList.size();
        params.d_dstEmbData = densificationTask.d_workingEmbDataset;
        params.dstEmbDim = densificationTask.globalEmbIdxEndExcl - densificationTask.globalEmbIdxBeginIncl;
        params.embDimToCopyBeginIncl = embDimBeginReal;
        params.embDimToCopyEndExcl = embDimEndReal;
        params.dstEmbOffset = embDimBeginReal - densificationTask.globalEmbIdxBeginIncl;
        params.d_copyTasks = densificationTask.d_copyTasks;
        params.numCopyTasks = densificationTask.numCopyTasks;

        if (densificationTask.numCopyTasks == 0)
        {
            continue;
        }
        constexpr int kBlockSize = 1024;
        int numTasks = densificationTask.numCopyTasks * compressibleEmbDim;
        int gridSize = (numTasks + kBlockSize - 1) / kBlockSize;
        densifyFromResQuantKernel<<<gridSize, kBlockSize>>>(params);
        CHECK_CUDA(cudaGetLastError());
        CHECK_CUDA(cudaDeviceSynchronize());
    }
}

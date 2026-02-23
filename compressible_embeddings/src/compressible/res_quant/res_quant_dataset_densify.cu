#include <algorithm>

#include "compressible/res_quant/res_quant_dataset.hpp"
#include "utils/util.hpp"

// ============================================================================
// Densification kernel
// ============================================================================

struct DensifyFromResQuantKernelParams
{
    // Centroid data
    const T_EMB* d_centroidEmb;
    size_t numCentroids;
    size_t globalEmbDim;

    // Per-doc data
    const int* d_centroidIdx;
    const T_RQ* d_residual;
    size_t numDocsTotal;
    size_t rqDim;

    // RQ config
    int numBitsPerDim;

    // Densification task
    int numDocsToDensify;
    T_EMB* d_workingEmbDataset;
    size_t embDimWorking;

    // Which compressible partition and its offset in the working index
    size_t compressibleEmbDimBegin;
    size_t compressibleEmbDimEnd;
    size_t embOffsetDst;

    CopyTask* d_copyTasks;
    int numCopyTasks;
};

__global__ void densifyFromResQuantKernel(DensifyFromResQuantKernelParams params)
{
    size_t tidx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t compressibleEmbDim = params.compressibleEmbDimEnd - params.compressibleEmbDimBegin;
    int copyIdx = tidx / compressibleEmbDim;
    int localEmbIdx = tidx % compressibleEmbDim;

    if (copyIdx < params.numCopyTasks)
    {
        CopyTask task = params.d_copyTasks[copyIdx];
        T_DOC_IDX docIdx = task.srcDocIdx;
        int toScoreIdx = task.dstDocIdx;

        int globalEmbIdx = localEmbIdx + params.compressibleEmbDimBegin;
        int centroidIdx = params.d_centroidIdx[docIdx];

        // Get centroid value and stdDev
        size_t centroidMemAddr
            = getMemAddrRowMajor(centroidIdx, globalEmbIdx * 2, params.numCentroids, params.globalEmbDim * 2);
        T_EMB centroid = params.d_centroidEmb[centroidMemAddr];
        float stdDev = static_cast<float>(params.d_centroidEmb[centroidMemAddr + 1]);

        // Get quantized residual and dequantize
        int rqIdx = getRqIdx(globalEmbIdx, params.numBitsPerDim, kBitsPerRqInt);
        size_t rqMemAddr = getMemAddrRowMajor(docIdx, rqIdx, params.numDocsTotal, params.rqDim);
        T_RQ quantRes = params.d_residual[rqMemAddr];
        float residual = dequantize(params.numBitsPerDim, kBitsPerRqInt, stdDev, quantRes, globalEmbIdx);

        // Reconstruct: centroid + residual
        T_EMB rst = static_cast<T_EMB>(static_cast<float>(centroid) + residual);

        // Write to working index
        size_t dstMemAddr = getMemAddrRowMajor(toScoreIdx,
                                               localEmbIdx + params.embOffsetDst,
                                               params.numDocsToDensify,
                                               params.embDimWorking);
        params.d_workingEmbDataset[dstMemAddr] = rst;
    }
}

// ============================================================================
// densifyCompressible
// ============================================================================

void ResQuantDataset::densifyCompressible(const DensificationTask& densificationTask) const
{
    size_t globalEmbDim = m_rqDim * kBitsPerRqInt / m_numBitsPerDim;

    for (const auto& compressibleConfig : m_compressiblePartitionConfigs)
    {
        // Compute the overlap between this compressible partition and the requested range.
        size_t embDimBeginReal
            = std::max(densificationTask.globalEmbIdxBeginIncl, compressibleConfig.getEmbDimBeginIncl());
        size_t embDimEndReal = std::min(densificationTask.globalEmbIdxEndExcl, compressibleConfig.getEmbDimEndExcl());

        if (embDimBeginReal >= embDimEndReal)
        {
            continue; // No overlap
        }

        size_t compressibleEmbDim = embDimEndReal - embDimBeginReal;

        DensifyFromResQuantKernelParams params;
        params.d_centroidEmb = m_d_centroidEmb.data();
        params.numCentroids = m_numCentroids;
        params.globalEmbDim = globalEmbDim;
        params.d_centroidIdx = m_d_centroidIdx.data();
        params.d_residual = m_d_residual.data();
        params.numDocsTotal = m_d_centroidIdx.getArraySize();
        params.rqDim = m_rqDim;
        params.numBitsPerDim = m_numBitsPerDim;
        params.numDocsToDensify = densificationTask.desiredDocIdxList.size();
        params.d_workingEmbDataset = densificationTask.d_workingEmbDataset;
        params.embDimWorking = densificationTask.globalEmbIdxEndExcl - densificationTask.globalEmbIdxBeginIncl;
        params.compressibleEmbDimBegin = embDimBeginReal;
        params.compressibleEmbDimEnd = embDimEndReal;
        params.embOffsetDst = embDimBeginReal - densificationTask.globalEmbIdxBeginIncl;
        params.d_copyTasks = densificationTask.d_copyTasks;
        params.numCopyTasks = densificationTask.numCopyTasks;

        if (densificationTask.numCopyTasks == 0)
        {
            continue;
        }
        constexpr size_t kBlockSize = 1024;
        size_t numTasks = densificationTask.numCopyTasks * compressibleEmbDim;
        size_t gridSize = (numTasks + kBlockSize - 1) / kBlockSize;
        densifyFromResQuantKernel<<<gridSize, kBlockSize>>>(params);
        CHECK_CUDA(cudaGetLastError());
        CHECK_CUDA(cudaDeviceSynchronize());
    }
}

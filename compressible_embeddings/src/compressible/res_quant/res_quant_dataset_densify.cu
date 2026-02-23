#include <algorithm>

#include "common/typedef.hpp"
#include "compressible/res_quant/res_quant_dataset.hpp"
#include "utils/util.hpp"
#include "common/const.hpp"
#include "compressible/res_quant/res_quant_utils.hpp"

/*
Please first read the comments in the resident/resident_emb_dataset_densify.cu file, we will use the same example to
explain how this works here. Also we assume there are 2048 centroids.

Take the same example as in the resident/resident_emb_dataset_densify.cu file, but let's slightly change it to - we want to 
densify the following dims: [64, 768] (704d)

This will transform into the following copies:
  1. Copy  CompressibleEmbDataset[13][64:128] to WorkingEmbDataset[1][0:64]
  2. Copy  CompressibleEmbDataset[ 9][64:128] to WorkingEmbDataset[3][0:64]
  3. Copy       ResidentEmbDataset[13][0:384] to WorkingEmbDataset[1][64:448]
  4. Copy       ResidentEmbDataset[ 9][0:384] to WorkingEmbDataset[3][64:448]
  5. Copy CompressibleEmbDataset[13][512:768] to WorkingEmbDataset[1][448:704]
  6. Copy CompressibleEmbDataset[ 9][512:768] to WorkingEmbDataset[3][448:704]

The kernel in this file will perform 1, 2, 5, 6, while 3 and 4 are performed in the resident/resident_emb_dataset_densify.cu.

Note that 1 + 2 are considered as the same partition will be performed together, so as 5 + 6. That's why we have the following loop:
```
for (const auto& compressibleConfig : m_compressiblePartitionConfigs)
```
*/

struct DensifyFromResQuantKernelParams
{
    // Centroid data
    const T_EMB* d_centroidEmb; // 2048 (numCentroids) x 1024 (globalEmbDim) x 2 (interleaved [emb, stdDev] per dim)
    int numCentroids; // 2048

    // RQ config
    int numBitsPerDim; // 2
    int rqDim; // 1024 (globalEmbDim) * 2 (numBitsPerDim) / 32 (kBitsPerRqInt) = 64

    // Source
    const int* d_srcCentroidIdx; // 10M (numDocs) x 1
    const T_RQ* h_srcResidual; // 10M (numDocs) x 64 (rqDim)
    int srcEmbDim; // 1024 (globalEmbDim)
    T_DOC_IDX srcNumDocs; // 10M (numDocs)

    // Destination
    T_EMB* d_dstEmbData; // 4 (dstNumDocs) x 704 (dstEmbDim)
    int dstEmbOffset; // first partition: 0, second partition: 448
    T_DOC_IDX dstNumDocs; // always 4
    int dstEmbDim; // always 768 - 64 = 704

    // Copy tasks
    int embDimToCopyBeginIncl; // always 64
    int embDimToCopyEndExcl; // always 768
    CopyTask* d_copyTasks; // [(srcDocIdx=13, dstDocIdx=1), (srcDocIdx=9, dstDocIdx=3)]
    int numCopyTasks; // 2
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

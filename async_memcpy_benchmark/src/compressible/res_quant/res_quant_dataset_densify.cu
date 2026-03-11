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
    int srcEmbOffset; // first partition: 64, second partition: 512
    T_DOC_IDX srcNumDocs; // 10M (numDocs)

    // Destination
    T_EMB* d_dstEmbData; // 4 (dstNumDocs) x 704 (dstEmbDim)
    int dstEmbOffset; // first partition: 0, second partition: 448
    T_DOC_IDX dstNumDocs; // always 4
    int dstEmbDim; // always 768 - 64 = 704

    // Copy tasks
    int embDimToCopy; // first partition: 128 - 64 = 64, second partition: 768 - 512 = 256
    CopyTask* d_copyTasks; // [(srcDocIdx=13, dstDocIdx=1), (srcDocIdx=9, dstDocIdx=3)]
    int numCopyTasks; // 2
};

__global__ void densifyFromResQuantKernel(DensifyFromResQuantKernelParams params)
{
    size_t tidx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    int copyIdx = tidx / params.embDimToCopy;
    int embIdxToCopy = tidx % params.embDimToCopy;

    if (copyIdx < params.numCopyTasks)
    {
        CopyTask task = params.d_copyTasks[copyIdx];

        int srcEmbIdx = embIdxToCopy + params.srcEmbOffset;
        int centroidIdx = params.d_srcCentroidIdx[task.srcDocIdx];

        // Get centroid value and stdDev
        size_t centroidMemAddr
            = getMemAddrRowMajor(centroidIdx, srcEmbIdx * 2, params.numCentroids, params.srcEmbDim * 2);
        T_EMB centroid = params.d_centroidEmb[centroidMemAddr];
        float stdDev = static_cast<float>(params.d_centroidEmb[centroidMemAddr + 1]);

        // Get quantized residual and dequantize
        int rqIdx = getRqIdx(srcEmbIdx, params.numBitsPerDim, kBitsPerRqInt);
        size_t rqMemAddr = getMemAddrRowMajor(task.srcDocIdx, rqIdx, params.srcNumDocs, params.rqDim);
        T_RQ quantRes = params.h_srcResidual[rqMemAddr];
        float residual = dequantize(params.numBitsPerDim, kBitsPerRqInt, stdDev, quantRes, srcEmbIdx);

        // Reconstruct: centroid + residual
        T_EMB rst = static_cast<T_EMB>(static_cast<float>(centroid) + residual);

        // Write to working index
        size_t dstMemAddr = getMemAddrRowMajor(task.dstDocIdx,
                                               embIdxToCopy + params.dstEmbOffset,
                                               params.dstNumDocs,
                                               params.dstEmbDim);
        params.d_dstEmbData[dstMemAddr] = rst;
    }
}

void ResQuantDataset::densify(const DensificationTask& task) const
{
    // -----------------
    // If there are no copy tasks, then just return.
    if (task.numCopyTasks == 0)
    {
        return;
    }

    for (const auto& partitionCfg : m_compressiblePartitionConfigs)
    {
        // -----------------
        // For example, if this partition has [64:192], but the desired emb dims are [128:256], then for this partition,
        // we should do:
        //   - begin = max(64, 128) = 128
        //   - end = min(192, 256) = 192
        int embDimBeginReal = std::max(task.globalEmbIdxBeginIncl, partitionCfg.getEmbDimBeginIncl());
        int embDimEndReal = std::min(task.globalEmbIdxEndExcl, partitionCfg.getEmbDimEndExcl());

        // -----------------
        // Nothing to copy for this partition if the following condition is met:
        if (embDimBeginReal >= embDimEndReal)
        {
            continue;
        }

        // -----------------
        // Prepare the parameters for the kernel.
        DensifyFromResQuantKernelParams params;
        {
            // --------
            // Centroid data
            params.d_centroidEmb = m_d_centroidEmb.data();
            params.numCentroids = m_numCentroids;
            params.rqDim = m_rqDim;
            params.numBitsPerDim = m_numBitsPerDim;

            // --------
            // Source
            params.srcEmbDim = m_embDim;
            params.d_srcCentroidIdx = m_d_centroidIdx.data();
            params.h_srcResidual = m_h_residual.data();
            params.srcNumDocs = m_d_centroidIdx.getArraySize();
            params.srcEmbOffset = embDimBeginReal;

            // --------
            // Destination
            params.dstNumDocs = task.desiredDocIdxList.size();
            params.d_dstEmbData = task.d_workingEmbDataset;
            params.dstEmbDim = task.globalEmbIdxEndExcl - task.globalEmbIdxBeginIncl;
            params.dstEmbOffset = embDimBeginReal - task.globalEmbIdxBeginIncl;

            // --------
            // Copy tasks
            params.embDimToCopy = embDimEndReal - embDimBeginReal;
            params.d_copyTasks = task.d_copyTasks;
            params.numCopyTasks = task.numCopyTasks;
        }

        // -----------------
        // Launch the kernel.
        constexpr int kBlockSize = 1024;
        int numTasks = task.numCopyTasks * params.embDimToCopy;
        int gridSize = (numTasks + kBlockSize - 1) / kBlockSize;
        densifyFromResQuantKernel<<<gridSize, kBlockSize, 0, m_cudaStreamRead.get()>>>(params);
        CHECK_CUDA(cudaGetLastError());
        CHECK_CUDA(cudaStreamSynchronize(m_cudaStreamRead.get()));
    }
}

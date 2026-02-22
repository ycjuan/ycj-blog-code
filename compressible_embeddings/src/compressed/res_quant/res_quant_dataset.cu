#include "common/memory_layout.hpp"
#include "compressed/res_quant/res_quant_dataset.hpp"
#include "utils/util.hpp"

#include <algorithm>
#include <iostream>
#include <limits>
#include <omp.h>
#include <sstream>

// ============================================================================
// Constructor
// ============================================================================

ResQuantDataset::ResQuantDataset(size_t numDocs,
                                 size_t globalEmbDim,
                                 size_t maxNumDocsInWorkingDataset,
                                 std::vector<CompressiblePartitionConfig> compressiblePartitionConfigs,
                                 size_t numBitsPerDim,
                                 const std::vector<std::vector<float>>& centroidEmbs,
                                 const std::vector<std::vector<float>>& centroidStdDevs)
    : CompressibleEmbDataset(numDocs, globalEmbDim, {}, maxNumDocsInWorkingDataset)
    , m_numCentroids(centroidEmbs.size())
    , m_numBitsPerDim(numBitsPerDim)
    , m_rqDim(getRqDim(globalEmbDim, numBitsPerDim))
    , m_compressiblePartitionConfigs(std::move(compressiblePartitionConfigs))
    , m_d_centroidEmb(m_numCentroids * globalEmbDim * 2, "m_centroidEmb")
    , m_d_centroidIdx(numDocs, "m_centroidIdx")
    , m_d_residual(numDocs * m_rqDim, "m_residual")
    , m_h_centroidEmb(m_numCentroids * globalEmbDim * 2, "m_centroidEmbHost")
    , m_h_centroidIdx(numDocs, "m_centroidIdxHost")
    , m_h_residual(numDocs * m_rqDim, "m_residualHost")
{
    if (kBitsPerRqInt % numBitsPerDim != 0)
    {
        std::ostringstream oss;
        oss << "kBitsPerRqInt (" << kBitsPerRqInt << ") must be divisible by numBitsPerDim (" << numBitsPerDim << ")";
        throw std::runtime_error(oss.str());
    }

    // -------------------------------------------------------------------------
    // Copy centroid embeddings to host buffer and device (interleaved [emb, stdDev] per dim)
    {
        for (size_t c = 0; c < m_numCentroids; c++)
        {
            for (size_t d = 0; d < globalEmbDim; d++)
            {
                size_t addr = getMemAddrRowMajor(c, d * 2, m_numCentroids, globalEmbDim * 2);
                m_h_centroidEmb.data()[addr] = static_cast<T_EMB>(centroidEmbs[c][d]);
                m_h_centroidEmb.data()[addr + 1] = static_cast<T_EMB>(centroidStdDevs[c][d]);
            }
        }
        CHECK_CUDA(cudaMemcpy(m_d_centroidEmb.data(),
                              m_h_centroidEmb.data(),
                              m_numCentroids * globalEmbDim * 2 * sizeof(T_EMB),
                              cudaMemcpyHostToDevice));
    }
}

// ============================================================================
// Getters
// ============================================================================

size_t ResQuantDataset::getNumCentroids() const { return m_numCentroids; }
size_t ResQuantDataset::getNumBitsPerDim() const { return m_numBitsPerDim; }
size_t ResQuantDataset::getRqDimPerDoc() const { return m_rqDim; }

// ============================================================================
// update
// ============================================================================

void ResQuantDataset::update(const std::vector<T_DOC_IDX>& docIdxList,
                             const std::vector<std::vector<T_EMB>>& emb2D,
                             const std::vector<int>& centroidIdxList)
{
    size_t globalEmbDim = m_rqDim * kBitsPerRqInt / m_numBitsPerDim;

#pragma omp parallel for schedule(static)
    for (size_t i = 0; i < docIdxList.size(); i++)
    {
        T_DOC_IDX docIdx = docIdxList.at(i);

        // // Centroid assignment: find nearest centroid by L2 distance
        // float bestDist = std::numeric_limits<float>::max();
        // int centroidIdx = 0;
        // for (size_t c = 0; c < m_numCentroids; ++c)
        // {
        //     float dist = 0.0f;
        //     for (size_t d = 0; d < globalEmbDim; ++d)
        //     {
        //         size_t addr = getMemAddrRowMajor(c, d * 2, m_numCentroids, globalEmbDim * 2);
        //         float centroidVal = static_cast<float>(m_centroidEmbHost.data()[addr]);
        //         float diff = static_cast<float>(emb2D.at(i).at(d)) - centroidVal;
        //         dist += diff * diff;
        //     }
        //     if (dist < bestDist)
        //     {
        //         bestDist = dist;
        //         centroidIdx = static_cast<int>(c);
        //     }
        // }

        int centroidIdx = centroidIdxList.at(i);

        // Store centroid index
        m_h_centroidIdx.data()[docIdx] = centroidIdx;

        // Compute and quantize residuals for all dimensions
        for (size_t embIdx = 0; embIdx < globalEmbDim; embIdx++)
        {
            size_t centroidAddr = getMemAddrRowMajor(centroidIdx, embIdx * 2, m_numCentroids, globalEmbDim * 2);
            float centroidVal = static_cast<float>(m_h_centroidEmb.data()[centroidAddr]);
            float stdDev = static_cast<float>(m_h_centroidEmb.data()[centroidAddr + 1]);
            float embVal = static_cast<float>(emb2D.at(i).at(embIdx));
            float residual = embVal - centroidVal;

            int rqIdx = getRqIdx(embIdx, m_numBitsPerDim, kBitsPerRqInt);
            size_t rqMemAddr = getMemAddrRowMajor(docIdx, rqIdx, m_h_residual.getArraySize() / m_rqDim, m_rqDim);
            quantize(m_numBitsPerDim, kBitsPerRqInt, stdDev, residual, m_h_residual.data()[rqMemAddr], embIdx);
        }
    }

    // Copy centroid indices to device
    CHECK_CUDA(cudaMemcpy(m_d_centroidIdx.data(),
                          m_h_centroidIdx.data(),
                          m_d_centroidIdx.getArraySizeInBytes(),
                          cudaMemcpyHostToDevice));

    // Copy residuals to device
    CHECK_CUDA(
        cudaMemcpy(m_d_residual.data(), m_h_residual.data(), m_d_residual.getArraySizeInBytes(), cudaMemcpyHostToDevice));
}

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
    T_DOC_IDX* d_docIdxList;
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
        params.d_docIdxList = densificationTask.d_docIdxList;
        params.numDocsToDensify = densificationTask.numDocsToDensify;
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

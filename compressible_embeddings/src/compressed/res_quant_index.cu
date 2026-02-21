#include "compressed/res_quant_index.hpp"
#include "utils/util.hpp"
#include "common/memory_layout.hpp"

#include <algorithm>
#include <sstream>
#include <iostream>

// ============================================================================
// Constructor
// ============================================================================

ResQuantIndex::ResQuantIndex(size_t numDocs,
                             size_t globalEmbDim,
                             size_t maxNumDocsInWorkingIndex,
                             std::vector<CompressedPartitionConfig> compressedPartitionConfigs,
                             size_t numBitsPerDim,
                             const std::vector<std::vector<float>>& centroidEmbs,
                             const std::vector<std::vector<float>>& centroidStdDevs)
    : CompressedEmbIndex(numDocs, globalEmbDim, {}, maxNumDocsInWorkingIndex)
    , m_numCentroids(centroidEmbs.size())
    , m_numBitsPerDim(numBitsPerDim)
    , m_rqDim(getRqDim(globalEmbDim, numBitsPerDim))
    , m_compressedPartitionConfigs(std::move(compressedPartitionConfigs))
    , m_centroidEmb(m_numCentroids * globalEmbDim * 2, "m_centroidEmb")
    , m_centroidIdx(numDocs, "m_centroidIdx")
    , m_residual(numDocs * m_rqDim, "m_residual")
    , m_centroidEmbHost(m_numCentroids * globalEmbDim * 2, "m_centroidEmbHost")
    , m_centroidIdxHost(numDocs, "m_centroidIdxHost")
    , m_residualHost(numDocs * m_rqDim, "m_residualHost")
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
                m_centroidEmbHost.data()[addr] = static_cast<T_EMB>(centroidEmbs[c][d]);
                m_centroidEmbHost.data()[addr + 1] = static_cast<T_EMB>(centroidStdDevs[c][d]);
            }
        }
        CHECK_CUDA(cudaMemcpy(m_centroidEmb.data(),
                              m_centroidEmbHost.data(),
                              m_numCentroids * globalEmbDim * 2 * sizeof(T_EMB),
                              cudaMemcpyHostToDevice));
    }

}

// ============================================================================
// Getters
// ============================================================================

size_t ResQuantIndex::getNumCentroids() const { return m_numCentroids; }
size_t ResQuantIndex::getNumBitsPerDim() const { return m_numBitsPerDim; }
size_t ResQuantIndex::getRqDimPerDoc() const { return m_rqDim; }

// ============================================================================
// update
// ============================================================================

void ResQuantIndex::update(const std::vector<T_DOC_IDX>& docIdxList,
                           const std::vector<std::vector<T_EMB>>& emb2D,
                           const std::vector<int>& centroidIdxList)
{
    if (docIdxList.size() != centroidIdxList.size())
    {
        std::ostringstream oss;
        oss << "docIdxList.size() (" << docIdxList.size() << ") != centroidIdxList.size() (" << centroidIdxList.size() << ")";
        throw std::runtime_error(oss.str());
    }

    size_t globalEmbDim = m_rqDim * kBitsPerRqInt / m_numBitsPerDim;

    for (size_t i = 0; i < docIdxList.size(); i++)
    {
        T_DOC_IDX docIdx = docIdxList[i];
        int centroidIdx = centroidIdxList[i];

        // Store centroid index
        m_centroidIdxHost.data()[docIdx] = centroidIdx;

        // Compute and quantize residuals for all dimensions
        for (size_t embIdx = 0; embIdx < globalEmbDim; embIdx++)
        {
            size_t centroidAddr = getMemAddrRowMajor(centroidIdx, embIdx * 2, m_numCentroids, globalEmbDim * 2);
            float centroidVal = static_cast<float>(m_centroidEmbHost.data()[centroidAddr]);
            float stdDev = static_cast<float>(m_centroidEmbHost.data()[centroidAddr + 1]);
            float embVal = static_cast<float>(emb2D[i][embIdx]);
            float residual = embVal - centroidVal;

            int rqIdx = getRqIdx(embIdx, m_numBitsPerDim, kBitsPerRqInt);
            size_t rqMemAddr = getMemAddrRowMajor(docIdx, rqIdx, m_residualHost.getArraySize() / m_rqDim, m_rqDim);
            quantize(m_numBitsPerDim, kBitsPerRqInt, stdDev, residual, m_residualHost.data()[rqMemAddr], embIdx);
        }
    }

    // Copy centroid indices to device
    CHECK_CUDA(cudaMemcpy(m_centroidIdx.data(),
                          m_centroidIdxHost.data(),
                          m_centroidIdx.getArraySizeInBytes(),
                          cudaMemcpyHostToDevice));

    // Copy residuals to device
    CHECK_CUDA(cudaMemcpy(m_residual.data(),
                          m_residualHost.data(),
                          m_residual.getArraySizeInBytes(),
                          cudaMemcpyHostToDevice));
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
    T_EMB* d_workingEmbIndex;
    size_t embDimWorking;

    // Which compressed partition and its offset in the working index
    size_t compressedEmbDimBegin;
    size_t compressedEmbDimEnd;
    size_t embOffsetDst;

    int8_t* hp_isCached;
};

__global__ void densifyFromResQuantKernel(DensifyFromResQuantKernelParams params)
{
    size_t tidx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t compressedEmbDim = params.compressedEmbDimEnd - params.compressedEmbDimBegin;
    int toScoreIdx = tidx / compressedEmbDim;
    int localEmbIdx = tidx % compressedEmbDim;

    if (toScoreIdx < params.numDocsToDensify)
    {
        if (params.hp_isCached[toScoreIdx])
        {
            return;
        }

        int globalEmbIdx = localEmbIdx + params.compressedEmbDimBegin;
        T_DOC_IDX docIdx = params.d_docIdxList[toScoreIdx];
        int centroidIdx = params.d_centroidIdx[docIdx];

        // Get centroid value and stdDev
        size_t centroidMemAddr = getMemAddrRowMajor(centroidIdx, globalEmbIdx * 2,
                                                     params.numCentroids, params.globalEmbDim * 2);
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
        params.d_workingEmbIndex[dstMemAddr] = rst;
    }
}

// ============================================================================
// densifyCompressed
// ============================================================================

void ResQuantIndex::densifyCompressed(const DensificationTask& densificationTask) const
{
    size_t globalEmbDim = m_rqDim * kBitsPerRqInt / m_numBitsPerDim;

    for (const auto& compressedConfig : m_compressedPartitionConfigs)
    {
        // Compute the overlap between this compressed partition and the requested range.
        size_t embDimBeginReal = std::max(densificationTask.globalEmbIdxBeginIncl, compressedConfig.getEmbDimBeginIncl());
        size_t embDimEndReal = std::min(densificationTask.globalEmbIdxEndExcl, compressedConfig.getEmbDimEndExcl());

        if (embDimBeginReal >= embDimEndReal)
        {
            continue; // No overlap
        }

        size_t compressedEmbDim = embDimEndReal - embDimBeginReal;

        DensifyFromResQuantKernelParams params;
        params.d_centroidEmb = m_centroidEmb.data();
        params.numCentroids = m_numCentroids;
        params.globalEmbDim = globalEmbDim;
        params.d_centroidIdx = m_centroidIdx.data();
        params.d_residual = m_residual.data();
        params.numDocsTotal = m_centroidIdx.getArraySize();
        params.rqDim = m_rqDim;
        params.numBitsPerDim = m_numBitsPerDim;
        params.d_docIdxList = densificationTask.d_docIdxList;
        params.numDocsToDensify = densificationTask.numDocsToDensify;
        params.d_workingEmbIndex = densificationTask.d_workingEmbIndex;
        params.embDimWorking = densificationTask.globalEmbIdxEndExcl - densificationTask.globalEmbIdxBeginIncl;
        params.compressedEmbDimBegin = embDimBeginReal;
        params.compressedEmbDimEnd = embDimEndReal;
        params.embOffsetDst = embDimBeginReal - densificationTask.globalEmbIdxBeginIncl;
        params.hp_isCached = densificationTask.hp_isCached;

        constexpr size_t kBlockSize = 1024;
        size_t numTasks = densificationTask.numDocsToDensify * compressedEmbDim;
        size_t gridSize = (numTasks + kBlockSize - 1) / kBlockSize;
        densifyFromResQuantKernel<<<gridSize, kBlockSize>>>(params);
        CHECK_CUDA(cudaGetLastError());
        CHECK_CUDA(cudaDeviceSynchronize());
    }
}


#include <algorithm>
#include <cstring>
#include <omp.h>
#include <sstream>

#include "common/memory_layout.hpp"
#include "compressible/res_quant/res_quant_dataset.hpp"
#include "utils/util.hpp"
#include "common/const.hpp"
#include "compressible/res_quant/res_quant_utils.hpp"

// ============================================================================
// Constructor
// ============================================================================

ResQuantDataset::ResQuantDataset(T_DOC_IDX numDocs,
                                 int globalEmbDim,
                                 std::vector<CompressiblePartitionConfig> compressiblePartitionConfigs,
                                 int numBitsPerDim,
                                 const std::vector<std::vector<float>>& centroidEmbs,
                                 const std::vector<std::vector<float>>& centroidStdDevs)
    : m_numDocs(numDocs)
    , m_embDim(globalEmbDim)
    , m_numCentroids(centroidEmbs.size())
    , m_numBitsPerDim(numBitsPerDim)
    , m_rqDim(getRqDim(globalEmbDim, numBitsPerDim))
    , m_compressiblePartitionConfigs(std::move(compressiblePartitionConfigs))
    , m_d_centroidEmb(m_numCentroids * globalEmbDim * 2, "m_d_centroidEmb")
    , m_d_centroidIdx(numDocs, "m_d_centroidIdx")
    , m_h_residual(numDocs * m_rqDim, "m_h_residual")
    , m_h_centroidEmb(m_numCentroids * globalEmbDim * 2, "m_h_centroidEmb")
    , m_h_docIdxChunk(kMaxUpdateBatchSize, "m_h_docIdxChunk")
    , m_h_centroidIdxChunk(kMaxUpdateBatchSize, "m_h_centroidIdxChunk")
    , m_h_residualChunk(kMaxUpdateBatchSize * m_rqDim, "m_h_residualChunk")
{
    // --------------------
    // Validate inputs
    if (kBitsPerRqInt % numBitsPerDim != 0)
    {
        std::ostringstream oss;
        oss << "kBitsPerRqInt (" << kBitsPerRqInt << ") must be divisible by numBitsPerDim (" << numBitsPerDim << ")";
        throw std::runtime_error(oss.str());
    }

    // --------------------
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

int ResQuantDataset::getNumCentroids() const { return m_numCentroids; }
int ResQuantDataset::getNumBitsPerDim() const { return m_numBitsPerDim; }
int ResQuantDataset::getRqDimPerDoc() const { return m_rqDim; }

// ============================================================================
// update kernel
// ============================================================================

__global__ void updateResQuantCentroidKernel(const T_DOC_IDX* h_docIdxChunk,
                                             const int* h_centroidIdxChunk,
                                             int numDocsInChunk,
                                             int* d_centroidIdx)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < numDocsInChunk)
    {
        d_centroidIdx[h_docIdxChunk[i]] = h_centroidIdxChunk[i];
    }
}

// ============================================================================
// update
// ============================================================================

void ResQuantDataset::update(const std::vector<T_DOC_IDX>& docIdxList,
                             const std::vector<std::vector<T_EMB>>& emb2D,
                             const std::vector<int>& centroidIdxList)
{
    int globalEmbDim = m_rqDim * kBitsPerRqInt / m_numBitsPerDim;
    int numDocs = (int)docIdxList.size();

    // Loop over docs chunk by chunk.
    for (int chunkBeginIncl = 0; chunkBeginIncl < numDocs; chunkBeginIncl += kMaxUpdateBatchSize)
    {
        // -----------
        // Handle corner case to get the end index, and calculate real number of docs in this chunk.
        int chunkEndExcl = std::min(chunkBeginIncl + kMaxUpdateBatchSize, numDocs);
        int numDocsInChunk = chunkEndExcl - chunkBeginIncl;

        // -----------
        // Zero out the residual chunk buffer (quantize uses |= so must start from 0)
        memset(m_h_residualChunk.data(), 0, numDocsInChunk * m_rqDim * sizeof(T_RQ));

        // -----------
        // Fill chunk buffers: docIdx, centroidIdx, and quantized residuals
#pragma omp parallel for schedule(static)
        for (int i = 0; i < numDocsInChunk; i++)
        {
            T_DOC_IDX docIdx = docIdxList.at(chunkBeginIncl + i);
            int centroidIdx = centroidIdxList.at(chunkBeginIncl + i);

            m_h_docIdxChunk.data()[i] = docIdx;
            m_h_centroidIdxChunk.data()[i] = centroidIdx;

            for (int embIdx = 0; embIdx < globalEmbDim; embIdx++)
            {
                size_t centroidAddr = getMemAddrRowMajor(centroidIdx, embIdx * 2, m_numCentroids, globalEmbDim * 2);
                float centroidVal = static_cast<float>(m_h_centroidEmb.data()[centroidAddr]);
                float stdDev = static_cast<float>(m_h_centroidEmb.data()[centroidAddr + 1]);
                float residual = static_cast<float>(emb2D.at(chunkBeginIncl + i).at(embIdx)) - centroidVal;

                int rqIdx = getRqIdx(embIdx, m_numBitsPerDim, kBitsPerRqInt);
                size_t rqMemAddr = getMemAddrRowMajor(i, rqIdx, numDocsInChunk, m_rqDim);
                quantize(m_numBitsPerDim, kBitsPerRqInt, stdDev, residual, m_h_residualChunk.data()[rqMemAddr], embIdx);
            }
        }

        // -----------
        // Scatter centroid indices to device via kernel
        constexpr int kBlockSize = 1024;
        int gridSize = (numDocsInChunk + kBlockSize - 1) / kBlockSize;
        updateResQuantCentroidKernel<<<gridSize, kBlockSize, 0, m_cudaStreamWrite.get()>>>(
            m_h_docIdxChunk.data(), m_h_centroidIdxChunk.data(), numDocsInChunk, m_d_centroidIdx.data());

        // -----------
        // Scatter residuals from chunk buffer to the full host residual buffer
        for (int i = 0; i < numDocsInChunk; i++)
        {
            T_DOC_IDX docIdx = m_h_docIdxChunk.data()[i];
            memcpy(m_h_residual.data() + (size_t)docIdx * m_rqDim,
                   m_h_residualChunk.data() + (size_t)i * m_rqDim,
                   m_rqDim * sizeof(T_RQ));
        }

        CHECK_CUDA(cudaStreamSynchronize(m_cudaStreamWrite.get()));
    }
}


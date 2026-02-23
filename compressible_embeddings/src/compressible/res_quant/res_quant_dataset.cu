#include "common/memory_layout.hpp"
#include "compressible/res_quant/res_quant_dataset.hpp"
#include "utils/util.hpp"

#include <iostream>
#include <omp.h>
#include <sstream>

// ============================================================================
// Constructor
// ============================================================================

ResQuantDataset::ResQuantDataset(T_DOC_IDX numDocs,
                                 int globalEmbDim,
                                 int maxNumDocsInWorkingDataset,
                                 std::vector<CompressiblePartitionConfig> compressiblePartitionConfigs,
                                 int numBitsPerDim,
                                 const std::vector<std::vector<float>>& centroidEmbs,
                                 const std::vector<std::vector<float>>& centroidStdDevs)
    : m_numCentroids(centroidEmbs.size())
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

int ResQuantDataset::getNumCentroids() const { return m_numCentroids; }
int ResQuantDataset::getNumBitsPerDim() const { return m_numBitsPerDim; }
int ResQuantDataset::getRqDimPerDoc() const { return m_rqDim; }

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

        // This is too slow with CPU. We will use GPU for nearest centroid assignment.
        // Before we have that, we temporarily assume the centroid index is already computed outside.
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
    CHECK_CUDA(cudaMemcpy(m_d_residual.data(),
                          m_h_residual.data(),
                          m_d_residual.getArraySizeInBytes(),
                          cudaMemcpyHostToDevice));
}


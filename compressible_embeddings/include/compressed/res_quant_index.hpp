#pragma once

#include <vector>

#include "compressed/compressed_emb_index.hpp"
#include "compressed/compressed_partition_config.hpp"
#include "compressed/residual_quantization.hpp"
#include "common/typedef.hpp"
#include "common/densification_task.hpp"
#include "utils/cuda_malloc_raii.hpp"

class ResQuantIndex : public CompressedEmbIndex
{
public:
    ResQuantIndex(size_t numDocs,
                  size_t globalEmbDim,
                  std::vector<ResidentPartitionConfig> residentIndexConfigs,
                  size_t maxNumDocsInWorkingIndex,
                  size_t numCentroids,
                  size_t numBitsPerDim);

    // Set centroid embeddings (typically called once after training).
    // centroidEmbs: numCentroids x globalEmbDim (the centroid mean values)
    // centroidStdDevs: numCentroids x globalEmbDim (per-dimension standard deviations)
    void setCentroids(const std::vector<std::vector<float>>& centroidEmbs,
                      const std::vector<std::vector<float>>& centroidStdDevs);

    // Update per-document data: centroid assignment and quantized residuals.
    // docIdxList: which documents to update
    // emb2D: full embeddings for each doc (used to compute residuals for compressed dimensions)
    // centroidIdxList: centroid assignment for each doc (parallel to docIdxList)
    void update(const std::vector<T_DOC_IDX>& docIdxList,
                const std::vector<std::vector<T_EMB>>& emb2D,
                const std::vector<int>& centroidIdxList);

    // Densify compressed partitions by reconstructing from centroid + dequantized residual.
    void densifyCompressed(const DensificationTask& densificationTask) const;

    // Getters
    size_t getNumCentroids() const;
    size_t getNumBitsPerDim() const;
    size_t getRqDimPerDoc() const;

private:
    // RQ configuration
    size_t m_numCentroids;
    size_t m_numBitsPerDim;
    size_t m_rqDim; // number of T_RQ elements per document

    // Compressed partition configs (complement of resident partitions)
    std::vector<CompressedPartitionConfig> m_compressedPartitionConfigs;

    // Centroid data on device: numCentroids x globalEmbDim x 2 (interleaved [emb, stdDev] per dim)
    CudaDeviceArray<T_EMB> m_centroidEmb;

    // Per-document data on device
    CudaDeviceArray<int> m_centroidIdx;   // numDocs x 1
    CudaDeviceArray<T_RQ> m_residual;     // numDocs x rqDim

    // Host buffers for updates
    CudaHostArray<int> m_centroidIdxHost;
    CudaHostArray<T_RQ> m_residualHost;
};

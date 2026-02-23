#pragma once

#include <vector>

#include "common/densification_task.hpp"
#include "common/typedef.hpp"
#include "compressible/compressible_partition_config.hpp"
#include "compressible/res_quant/res_quant_utils.hpp"
#include "utils/cuda_raii.hpp"

class ResQuantDataset
{
public:

    ResQuantDataset(T_DOC_IDX numDocs,
                    int globalEmbDim,
                    std::vector<CompressiblePartitionConfig> compressiblePartitionConfigs,
                    int numBitsPerDim,
                    const std::vector<std::vector<float>>& centroidEmbs,
                    const std::vector<std::vector<float>>& centroidStdDevs);

    // Update per-document data: centroid assignment and quantized residuals.
    void update(const std::vector<T_DOC_IDX>& docIdxList,
                const std::vector<std::vector<T_EMB>>& emb2D,
                const std::vector<int>& centroidIdxList);

    // Densify compressible partitions by reconstructing from centroid + dequantized residual.
    void densifyCompressible(const DensificationTask& densificationTask) const;

    // Getters
    int getNumCentroids() const;
    int getNumBitsPerDim() const;
    int getRqDimPerDoc() const;

private:
    // RQ configuration
    int m_numCentroids;
    int m_numBitsPerDim; // How many bits we use to quantize each dimension
    int m_rqDim; // This equals to globalEmbDim * numBitsPerDim / kBitsPerRqInt

    // Compressible partition configs (complement of resident partitions)
    std::vector<CompressiblePartitionConfig> m_compressiblePartitionConfigs;

    // Centroid data on device: numCentroids x globalEmbDim x 2 (interleaved [emb, stdDev] per dim)
    CudaDeviceArray<T_EMB> m_d_centroidEmb;

    // Per-document data on device
    CudaDeviceArray<int> m_d_centroidIdx; // numDocs x 1
    CudaDeviceArray<T_RQ> m_d_residual; // numDocs x rqDim

    // Host-side copy of centroid embeddings (needed for computing residuals in update)
    CudaHostArray<T_EMB> m_h_centroidEmb;

    // Host buffers for updates
    CudaHostArray<int> m_h_centroidIdx;
    CudaHostArray<T_RQ> m_h_residual;
};

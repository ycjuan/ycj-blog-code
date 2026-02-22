#pragma once

#include <cuda_bf16.h>
#include <vector>

#include "common/typedef.hpp"
#include "common/densification_task.hpp"
#include "resident/resident_partition_config.hpp"
#include "utils/cuda_malloc_raii.hpp"

class ResidentEmbDataset
{
public:
    // Constructor
    ResidentEmbDataset(size_t numDocs, ResidentPartitionConfig residentPartitionConfig);

    // Getters
    T_EMB* data() const;
    ResidentPartitionConfig getResidentPartitionConfig() const;

    // Update
    void update(const std::vector<T_DOC_IDX>& docIdxList, const std::vector<std::vector<T_EMB>>& emb2D);

    // Densification
    void densify(const DensificationTask& densificationTask) const;

private:
    // Constants
    static constexpr size_t kMaxUpdateBatchSize = 10000;

    // Meta data
    size_t m_numDocs;
    ResidentPartitionConfig m_residentPartitionConfig;

    // Index
    CudaDeviceArray<T_EMB> m_d_embData;

    // Update buffer
    CudaHostArray<T_DOC_IDX> m_h_docIdxChunk;
    CudaHostArray<T_EMB> m_h_embDataChunk;

    // Stream
    CudaStream m_cudaStreamRead;
    CudaStream m_cudaStreamWrite;
};

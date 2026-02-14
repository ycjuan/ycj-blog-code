#pragma once

#include <cuda_bf16.h>
#include <vector>

#include "common/typedef.hpp"
#include "compressed/compressed_partition_config.hpp"
#include "resident/resident_emb_index.hpp"
#include "resident/resident_partition_config.hpp"
#include "working/working_emb_index.hpp"
#include "utils/cuda_malloc_raii.hpp"

class EmbIndexManager
{
public:
    EmbIndexManager(size_t numDocs,
                    size_t totalEmbDim,
                    std::vector<ResidentPartitionConfig> residentPartitionConfigs,
                    size_t maxWorkingSetSize);

    void update(const std::vector<T_DOC_IDX>& docIdxList, const std::vector<std::vector<T_EMB>>& emb2D);

    const WorkingEmbIndex& densify(const std::vector<T_DOC_IDX>& docIdxList,
                                   size_t embIdxBeginIncl,
                                   size_t embIdxEndExcl,
                                   MemLayout memLayout);

protected:
    // General meta data
    size_t m_numDocs;
    size_t m_totalEmbDim;
    size_t m_maxWorkingSetSize;

    // Resident index
    std::vector<ResidentEmbIndex> m_residentEmbIndices;

    // Working set index
    WorkingEmbIndex m_workingEmbIndex;

    // Densification
    CudaDeviceArray<T_DOC_IDX> m_docIdxListToCopy;

    // Cuda stream
    cudaStream_t m_cudaStreamRead;
    cudaStream_t m_cudaStreamWrite;

};

/*
https://en.wikipedia.org/wiki/Resident_set_size
*/
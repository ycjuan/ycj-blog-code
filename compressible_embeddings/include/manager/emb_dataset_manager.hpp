#pragma once

#include <cuda_bf16.h>
#include <iomanip>
#include <iostream>
#include <vector>
#include <unordered_map>

#include "common/typedef.hpp"
#include "compressed/res_quant_dataset.hpp"
#include "resident/resident_emb_dataset.hpp"
#include "resident/resident_partition_config.hpp"
#include "working/working_emb_dataset.hpp"
#include "utils/cuda_malloc_raii.hpp"

struct TimeRecord
{
    // cache() segments
    float cacheFirstScanMs = 0.0f;
    float cacheSecondScanMs = 0.0f;
    float cacheReassignMs = 0.0f;

    // densify() segments
    float densifyCacheMs = 0.0f;
    float densifyMemcpyH2DMs = 0.0f;
    std::vector<float> densifyResidentPartitionMs;
    float densifyCompressedMs = 0.0f;

    float cacheTotalMs = 0.0f;
    float densifyTotalMs = 0.0f;

    void print() const
    {
        std::cout << std::fixed << std::setprecision(3)
                  << "  [densify] total: " << densifyTotalMs << " ms\n"
                  << "  [densify] cache: " << densifyCacheMs << " ms (total: " << cacheTotalMs << " ms)\n"
                  << "    [cache] first scan: " << cacheFirstScanMs << " ms\n"
                  << "    [cache] second scan: " << cacheSecondScanMs << " ms\n"
                  << "    [cache] reassign: " << cacheReassignMs << " ms\n"
                  << "  [densify] cudaMemcpy docIdxList H2D: " << densifyMemcpyH2DMs << " ms\n";
        for (size_t i = 0; i < densifyResidentPartitionMs.size(); ++i)
        {
            std::cout << "  [densify] residentPartition[" << i << "]: " << densifyResidentPartitionMs[i] << " ms\n";
        }
        std::cout << "  [densify] densifyCompressed: " << densifyCompressedMs << " ms\n";
    }
};

class EmbDatasetManager
{
public:
    EmbDatasetManager(size_t numDocs,
                    size_t globalEmbDim,
                    std::vector<ResidentPartitionConfig> residentPartitionConfigs,
                    size_t maxNumWorkingDocs,
                    size_t numBitsPerDim,
                    const std::vector<std::vector<float>>& centroidEmbs,
                    const std::vector<std::vector<float>>& centroidStdDevs);

    void update(const std::vector<T_DOC_IDX>& docIdxList, const std::vector<std::vector<T_EMB>>& emb2D);

    const WorkingEmbDataset& densify(std::vector<T_DOC_IDX>& docIdxList,
                                   size_t globalEmbIdxBeginIncl,
                                   size_t globalEmbIdxEndExcl,
                                   MemLayout memLayout);

    const TimeRecord& getLastTimeRecord() const { return m_lastTimeRecord; }

protected:
    // General meta data
    size_t m_numDocs;
    size_t m_totalEmbDim;
    size_t m_maxNumWorkingDocs;

    // Resident datasets
    std::vector<ResidentEmbDataset> m_residentEmbDatasets;

    // Compressed dataset (residual quantization)
    ResQuantDataset m_resQuantDataset;

    // Working set dataset
    WorkingEmbDataset m_workingEmbDataset;

    // Densification
    CudaDeviceArray<T_DOC_IDX> m_docIdxListToDensify;

    // Centroid embeddings (host, for nearest-centroid assignment)
    std::vector<std::vector<float>> m_centroidEmbs;

    // Cuda stream
    cudaStream_t m_cudaStreamRead;
    cudaStream_t m_cudaStreamWrite;

    // For caching
    std::unordered_map<T_DOC_IDX, T_DOC_IDX> m_cachedDocIdxToWorkingIdx;
    std::vector<T_DOC_IDX> m_cachedWorkingIdxToDocIdx;
    void cache(std::vector<T_DOC_IDX>& docIdxList);
    CudaHostArray<int8_t> m_hp_isCached;

    // Time record
    TimeRecord m_lastTimeRecord;
};

/*
https://en.wikipedia.org/wiki/Resident_set_size
*/
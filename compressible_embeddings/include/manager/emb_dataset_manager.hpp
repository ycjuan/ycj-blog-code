#pragma once

#include <cuda_bf16.h>
#include <unordered_map>
#include <vector>

#include "common/densification_task.hpp"
#include "common/typedef.hpp"
#include "compressible/res_quant/res_quant_dataset.hpp"
#include "resident/resident_emb_dataset.hpp"
#include "resident/resident_partition_config.hpp"
#include "utils/cuda_raii.hpp"
#include "working/working_emb_dataset.hpp"

struct TimeRecord
{
    // cache() segments
    float cacheFirstScanMs = 0.0f;
    float cacheSecondScanMs = 0.0f;
    float cacheReassignMs = 0.0f;
    
    // densify() segments
    float densifyTotalMs = 0.0f;
    float densifyCacheMs = 0.0f;
    float densifyCopyTasksMs = 0.0f;
    std::vector<float> densifyResidentPartitionMs;
    float densifyCompressibleMs = 0.0f;

    // count
    int count = 0;

    void print() const;
};

class EmbDatasetManager
{
public:

    EmbDatasetManager(T_DOC_IDX numDocs,
                      int globalEmbDim,
                      std::vector<ResidentPartitionConfig> residentPartitionConfigs,
                      int maxNumWorkingDocs,
                      int numBitsPerDim,
                      const std::vector<std::vector<float>>& centroidEmbs,
                      const std::vector<std::vector<float>>& centroidStdDevs);

    void update(const std::vector<T_DOC_IDX>& docIdxList,
                const std::vector<std::vector<T_EMB>>& emb2D,
                const std::vector<int>& centroidIdxList);

    const WorkingEmbDataset& densify(DensificationTask& task);

    TimeRecord getLastTimeRecordAndReset();

protected:
    // General meta data
    T_DOC_IDX m_numDocs;
    int m_totalEmbDim;
    int m_maxNumWorkingDocs;

    // Resident datasets
    std::vector<ResidentEmbDataset> m_residentEmbDatasets;

    // Compressible dataset (residual quantization)
    ResQuantDataset m_resQuantDataset;

    // Working set dataset
    WorkingEmbDataset m_workingEmbDataset;

    // Cuda stream
    cudaStream_t m_cudaStreamRead;
    cudaStream_t m_cudaStreamWrite;

    // For caching
    std::unordered_map<T_DOC_IDX, int> m_currDocIdxToWorkingIdx;
    std::vector<T_DOC_IDX> m_currDocIdxListInWorkingDataset;
    void cache(std::vector<T_DOC_IDX>& docIdxList);
    CudaHostArray<CopyTask> m_h_copyTasks;
    CudaDeviceArray<CopyTask> m_d_copyTasks;
    int m_numCopyTasks = 0;

    // Time record
    TimeRecord m_lastTimeRecord;
};
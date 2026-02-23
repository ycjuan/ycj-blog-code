#include <algorithm>
#include <iomanip>
#include <iostream>

#include "common/const.hpp"
#include "common/typedef.hpp"
#include "manager/emb_dataset_manager.hpp"
#include "utils/util.hpp"

void TimeRecord::print() const
{
    float n = static_cast<float>(count);
    std::cout << std::fixed << std::setprecision(3) << "[densify] count: " << count << "\n"
              << "[densify] total: " << densifyTotalMs / n << " ms avg\n"
              << "[densify] total - cache: " << (densifyTotalMs - densifyCacheMs) / n << " ms avg\n"
              << "[densify] cache: " << densifyCacheMs / n << " ms avg\n"
              << "          [cache] first scan: " << cacheFirstScanMs / n << " ms avg\n"
              << "          [cache] second scan: " << cacheSecondScanMs / n << " ms avg\n"
              << "          [cache] reassign: " << cacheReassignMs / n << " ms avg\n"
              << "[densify] copy tasks: " << densifyCopyTasksMs / n << " ms avg\n";
    for (size_t i = 0; i < densifyResidentPartitionMs.size(); ++i)
    {
        std::cout << "[densify] residentPartition[" << i << "]: " << densifyResidentPartitionMs[i] / n << " ms avg\n";
    }
    std::cout << "[densify] densifyCompressible: " << densifyCompressibleMs / n << " ms avg\n";
}

TimeRecord EmbDatasetManager::getLastTimeRecordAndReset()
{
    TimeRecord record = m_lastTimeRecord;
    m_lastTimeRecord = TimeRecord {};
    return record;
}

EmbDatasetManager::EmbDatasetManager(T_DOC_IDX numDocs,
                                     int totalEmbDim,
                                     std::vector<ResidentPartitionConfig> residentPartitionConfigs,
                                     int maxNumWorkingDocs,
                                     int numBitsPerDim,
                                     const std::vector<std::vector<float>>& centroidEmbs,
                                     const std::vector<std::vector<float>>& centroidStdDevs)
    : m_numDocs(numDocs)
    , m_totalEmbDim(totalEmbDim)
    , m_maxNumWorkingDocs(maxNumWorkingDocs)
    , m_resQuantDataset(numDocs,
                        totalEmbDim,
                        maxNumWorkingDocs,
                        findCompressiblePartitionConfigs(residentPartitionConfigs, totalEmbDim),
                        numBitsPerDim,
                        centroidEmbs,
                        centroidStdDevs)
    , m_workingEmbDataset(maxNumWorkingDocs, totalEmbDim)
    , m_h_copyTasks(maxNumWorkingDocs, "m_h_copyTasks")
    , m_d_copyTasks(maxNumWorkingDocs, "m_d_copyTasks")
    , m_currDocIdxListInWorkingDataset(maxNumWorkingDocs, kInvalidDocIdx)
{
    // --------------
    // We don't really need this sort, but we just do it for convenience.
    std::sort(residentPartitionConfigs.begin(), residentPartitionConfigs.end());

    // --------------
    // Confirm the resident partitions are disjoint in embDim space.
    for (size_t i = 1; i < residentPartitionConfigs.size(); ++i)
    {
        if (residentPartitionConfigs.at(i - 1).getEmbDimEndExcl() > residentPartitionConfigs.at(i).getEmbDimBeginIncl())
        {
            std::ostringstream oss;
            oss << "Resident partition configs have overlapping embedding dimension ranges: "
                << residentPartitionConfigs.at(i - 1).getEmbDimEndExcl() << " > "
                << residentPartitionConfigs.at(i).getEmbDimBeginIncl();
            throw std::runtime_error(oss.str());
        }
    }

    // --------------
    // Initialize the resident datasets.
    for (const auto& residentPartitionConfig : residentPartitionConfigs)
    {
        m_residentEmbDatasets.push_back(ResidentEmbDataset(numDocs, residentPartitionConfig));
    }
}

void EmbDatasetManager::update(const std::vector<T_DOC_IDX>& docIdxList,
                               const std::vector<std::vector<T_EMB>>& emb2D,
                               const std::vector<int>& centroidIdxList)
{
    // --------------
    // Some input sanity checks.
    if (docIdxList.size() > m_numDocs)
    {
        std::ostringstream oss;
        oss << "docIdxList.size() > m_numDocs: " << docIdxList.size() << " > " << m_numDocs;
        throw std::runtime_error(oss.str());
    }

    // --------------
    // Update the resident datasets.
    for (size_t residentPartitionIdx = 0; residentPartitionIdx < m_residentEmbDatasets.size(); ++residentPartitionIdx)
    {
        auto& embDataset = m_residentEmbDatasets[residentPartitionIdx];
        embDataset.update(docIdxList, emb2D);
    }

    // --------------
    // Update the compressible dataset.
    m_resQuantDataset.update(docIdxList, emb2D, centroidIdxList);
}

const WorkingEmbDataset& EmbDatasetManager::densify(DensificationTask& task)
{
    // --------------
    // Input sanity checks.
    if (task.desiredDocIdxList.size() > m_maxNumWorkingDocs)
    {
        std::ostringstream oss;
        oss << "docIdxList.size() > m_maxNumWorkingDocs: " << task.desiredDocIdxList.size() << " > " << m_maxNumWorkingDocs;
        throw std::runtime_error(oss.str());
    }

    // --------------
    // Prepare timers
    Timer e2eTimer;
    {
        m_lastTimeRecord.count++;
        e2eTimer.tic();
    }

    // ------------
    // Cache the docIdxList.
    {
        Timer timer;
        timer.tic();
        cache(task.desiredDocIdxList);
        m_lastTimeRecord.densifyCacheMs += timer.tocMs();
    }

    // ------------
    // Copy the copy tasks to the device.
    {
        Timer timer;
        timer.tic();
        CHECK_CUDA(cudaMemcpy(m_d_copyTasks.data(), m_h_copyTasks.data(), m_numCopyTasks * sizeof(CopyTask), cudaMemcpyHostToDevice));
        m_lastTimeRecord.densifyCopyTasksMs += timer.tocMs();
    }

    // --------------
    // Set the working dataset properties.
    {
        m_workingEmbDataset.setMemLayout(task.memLayout);
        m_workingEmbDataset.setEmbDimBeginIncl(task.globalEmbIdxBeginIncl);
        m_workingEmbDataset.setEmbDimEndExcl(task.globalEmbIdxEndExcl);
    }

    // --------------
    // Fill in the manager-side fields of the densification task.
    {
        task.numCopyTasks = m_numCopyTasks;
        task.d_workingEmbDataset = m_workingEmbDataset.data();
        task.d_copyTasks = m_d_copyTasks.data();
    }

    // --------------
    // Densify the resident datasets.
    {
        m_lastTimeRecord.densifyResidentPartitionMs.resize(m_residentEmbDatasets.size());
        for (size_t residentPartitionIdx = 0; residentPartitionIdx < m_residentEmbDatasets.size();
             ++residentPartitionIdx)
        {
            Timer timer;
            timer.tic();
            const auto& embDataset = m_residentEmbDatasets[residentPartitionIdx];
            embDataset.densify(task);
            m_lastTimeRecord.densifyResidentPartitionMs[residentPartitionIdx] += timer.tocMs();
        }
    }
    // --------------
    // Densify the compressible dataset.
    {
        Timer timer;
        timer.tic();
        m_resQuantDataset.densifyCompressible(task);
        m_lastTimeRecord.densifyCompressibleMs += timer.tocMs();
    }

    // --------------
    // Record the end time.
    {
        m_lastTimeRecord.densifyTotalMs += e2eTimer.tocMs();
    }

    return m_workingEmbDataset;
}


#include <algorithm>
#include <iomanip>
#include <iostream>
#include <stack>
#include <unordered_set>

#include "common/typedef.hpp"
#include "manager/emb_dataset_manager.hpp"
#include "utils/util.hpp"

namespace // Anonymous namespace to avoid polluting the global namespace.
{
constexpr T_DOC_IDX kInvalidDocIdx = -1; // Use -1 to indicate an invalid index in caching.
constexpr bool kDebug = false; // Set to true to perform some slow checks.
}

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
              << "[densify] cudaMemcpy docIdxList H2D: " << densifyMemcpyH2DMs / n << " ms avg\n";
    for (size_t i = 0; i < densifyResidentPartitionMs.size(); ++i)
    {
        std::cout << "[densify] residentPartition[" << i << "]: " << densifyResidentPartitionMs[i] / n << " ms avg\n";
    }
    std::cout << "[densify] densifyCompressed: " << densifyCompressedMs / n << " ms avg\n";
}

TimeRecord EmbDatasetManager::getLastTimeRecordAndReset()
{
    TimeRecord record = m_lastTimeRecord;
    m_lastTimeRecord = TimeRecord {};
    return record;
}

EmbDatasetManager::EmbDatasetManager(size_t numDocs,
                                     size_t totalEmbDim,
                                     std::vector<ResidentPartitionConfig> residentPartitionConfigs,
                                     size_t maxNumWorkingDocs,
                                     size_t numBitsPerDim,
                                     const std::vector<std::vector<float>>& centroidEmbs,
                                     const std::vector<std::vector<float>>& centroidStdDevs)
    : m_numDocs(numDocs)
    , m_totalEmbDim(totalEmbDim)
    , m_maxNumWorkingDocs(maxNumWorkingDocs)
    , m_resQuantDataset(numDocs,
                        totalEmbDim,
                        maxNumWorkingDocs,
                        findCompressedPartitionConfigs(residentPartitionConfigs, totalEmbDim),
                        numBitsPerDim,
                        centroidEmbs,
                        centroidStdDevs)
    , m_workingEmbDataset(maxNumWorkingDocs, totalEmbDim)
    , m_docIdxListToDensify(maxNumWorkingDocs, "m_docIdxListToDensify")
    , m_hp_isCached(maxNumWorkingDocs, "m_hp_isCached")
    , m_cachedWorkingIdxToDocIdx(maxNumWorkingDocs, kInvalidDocIdx)
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

void EmbDatasetManager::update(const std::vector<T_DOC_IDX>& docIdxList, const std::vector<std::vector<T_EMB>>& emb2D)
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
    // Update the compressed dataset.
    m_resQuantDataset.update(docIdxList, emb2D);
}

const WorkingEmbDataset& EmbDatasetManager::densify(std::vector<T_DOC_IDX>& docIdxList,
                                                    size_t embIdxBeginIncl,
                                                    size_t embIdxEndExcl,
                                                    MemLayout memLayout)
{
    // --------------
    // Input sanity checks.
    if (docIdxList.size() > m_maxNumWorkingDocs)
    {
        std::ostringstream oss;
        oss << "docIdxList.size() > m_maxNumWorkingDocs: " << docIdxList.size() << " > " << m_maxNumWorkingDocs;
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
        cache(docIdxList);
        m_lastTimeRecord.densifyCacheMs += timer.tocMs();
    }

    // --------------
    // Copy the docIdxList to the device.
    {
        Timer timer;
        timer.tic();
        CHECK_CUDA(cudaMemcpy(m_docIdxListToDensify.data(),
                              docIdxList.data(),
                              docIdxList.size() * sizeof(T_DOC_IDX),
                              cudaMemcpyHostToDevice));
        m_lastTimeRecord.densifyMemcpyH2DMs += timer.tocMs();
    }

    // --------------
    // Set the working dataset properties.
    {
        m_workingEmbDataset.setMemLayout(memLayout);
        m_workingEmbDataset.setEmbDimBeginIncl(embIdxBeginIncl);
        m_workingEmbDataset.setEmbDimEndExcl(embIdxEndExcl);
    }

    // --------------
    // Prepare the densification task.
    DensificationTask densificationTask;
    {
        densificationTask.numDocsToDensify = docIdxList.size();
        densificationTask.globalEmbIdxBeginIncl = embIdxBeginIncl;
        densificationTask.globalEmbIdxEndExcl = embIdxEndExcl;
        densificationTask.d_workingEmbDataset = m_workingEmbDataset.data();
        densificationTask.d_docIdxList = m_docIdxListToDensify.data();
        densificationTask.hp_isCached = m_hp_isCached.data();
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
            embDataset.densify(densificationTask);
            m_lastTimeRecord.densifyResidentPartitionMs[residentPartitionIdx] += timer.tocMs();
        }
    }
    // --------------
    // Densify the compressed dataset.
    {
        Timer timer;
        timer.tic();
        m_resQuantDataset.densifyCompressed(densificationTask);
        m_lastTimeRecord.densifyCompressedMs += timer.tocMs();
    }

    // --------------
    // Record the end time.
    {
        m_lastTimeRecord.densifyTotalMs += e2eTimer.tocMs();
    }

    return m_workingEmbDataset;
}

void EmbDatasetManager::cache(std::vector<T_DOC_IDX>& docIdxList)
{
    // ------------
    // Verify docIdxList is unique (only run in debug mode)
    if (kDebug)
    {
        std::unordered_set<T_DOC_IDX> seen;
        for (T_DOC_IDX docIdx : docIdxList)
        {
            if (!seen.insert(docIdx).second)
            {
                std::ostringstream oss;
                oss << "Duplicate docIdx in docIdxList: " << docIdx;
                throw std::runtime_error(oss.str());
            }
        }
    }

    // ------------
    // First scan to find the cached working indices.
    std::vector<T_DOC_IDX> reorderedDocIdxList(docIdxList.size(), kInvalidDocIdx);
    std::stack<T_DOC_IDX> uncachedDocIdxList;
    int cntCached = 0;
    int cntUncached = 0;
    {
        Timer timer;
        timer.tic();
        for (T_DOC_IDX workingIdx = 0; workingIdx < docIdxList.size(); ++workingIdx)
        {
            T_DOC_IDX docIdx = docIdxList.at(workingIdx);
            auto it = m_cachedDocIdxToWorkingIdx.find(docIdx);
            if (it != m_cachedDocIdxToWorkingIdx.end() && it->second < docIdxList.size())
            {
                T_DOC_IDX cachedWorkingIdx = it->second;
                reorderedDocIdxList.at(cachedWorkingIdx) = docIdx;
                m_hp_isCached.data()[cachedWorkingIdx] = 1;
                cntCached++;
            }
            else
            {
                // Very important: if the cached working index is larger than the docIdxList.size(),
                //                 it is still considered as uncached.
                uncachedDocIdxList.push(docIdx);
                cntUncached++;
            }
        }
        m_lastTimeRecord.cacheFirstScanMs += timer.tocMs();
    }

    // ------------
    // Verify no two docIdx map to the same cachedWorkingIdx. (only run in debug mode)
    if (kDebug)
    {
        std::unordered_set<T_DOC_IDX> seenWorkingIdx;
        for (T_DOC_IDX docIdx : docIdxList)
        {
            auto it = m_cachedDocIdxToWorkingIdx.find(docIdx);
            if (it != m_cachedDocIdxToWorkingIdx.end() && it->second < docIdxList.size())
            {
                if (!seenWorkingIdx.insert(it->second).second)
                {
                    std::ostringstream oss;
                    oss << "Two docIdx values map to same cachedWorkingIdx: " << it->second << " (docIdx=" << docIdx
                        << ")";
                    throw std::runtime_error(oss.str());
                }
            }
        }
    }

    // ------------
    // Second scan to put the uncached doc indices to the reorderedDocIdxList.
    {
        Timer timer;
        timer.tic();
        for (T_DOC_IDX workingIdx = 0; workingIdx < docIdxList.size(); ++workingIdx)
        {
            if (reorderedDocIdxList.at(workingIdx) == kInvalidDocIdx)
            {
                cntUncached--;
                T_DOC_IDX uncachedDocIdx = uncachedDocIdxList.top();
                reorderedDocIdxList.at(workingIdx) = uncachedDocIdx;
                uncachedDocIdxList.pop();
                T_DOC_IDX oldDocIdx = m_cachedWorkingIdxToDocIdx.at(workingIdx);
                if (oldDocIdx != kInvalidDocIdx)
                {
                    m_cachedDocIdxToWorkingIdx.erase(oldDocIdx);
                }
                m_cachedDocIdxToWorkingIdx[uncachedDocIdx] = workingIdx;
                m_cachedWorkingIdxToDocIdx.at(workingIdx) = uncachedDocIdx;
                m_hp_isCached.data()[workingIdx] = 0;
            }
            else
            {
                cntCached--;
            }
        }
        m_lastTimeRecord.cacheSecondScanMs += timer.tocMs();
    }

    // ------------
    // Verifications
    {
        if (!uncachedDocIdxList.empty())
        {
            std::ostringstream oss;
            oss << "Uncached doc indices are not empty: " << uncachedDocIdxList.size();
            throw std::runtime_error(oss.str());
        }
        if (cntCached != 0)
        {
            std::ostringstream oss;
            oss << "cntCached != 0: " << cntCached;
            throw std::runtime_error(oss.str());
        }
        if (cntUncached != 0)
        {
            std::ostringstream oss;
            oss << "cntUncached != 0: " << cntUncached;
            throw std::runtime_error(oss.str());
        }
    }

    // ------------
    // Reassign the reorderedDocIdxList to the docIdxList.
    {
        Timer timer;
        timer.tic();
        docIdxList = reorderedDocIdxList;
        m_lastTimeRecord.cacheReassignMs += timer.tocMs();
    }
}
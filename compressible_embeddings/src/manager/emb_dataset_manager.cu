#include <algorithm>
#include <limits>
#include <queue>
#include <unordered_set>

#include "common/typedef.hpp"
#include "manager/emb_dataset_manager.hpp"
#include "utils/util.hpp"

namespace // Anonymous namespace to avoid polluting the global namespace.
{
constexpr T_DOC_IDX kInvalidDocIdx = -1;
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
    , m_compressedPartitionConfigs(findCompressedPartitionConfigs(residentPartitionConfigs, totalEmbDim))
    , m_resQuantDataset(numDocs,
                        totalEmbDim,
                        maxNumWorkingDocs,
                        m_compressedPartitionConfigs,
                        numBitsPerDim,
                        centroidEmbs,
                        centroidStdDevs)
    , m_workingEmbDataset(maxNumWorkingDocs, totalEmbDim)
    , m_docIdxListToDensify(maxNumWorkingDocs, "m_docIdxListToDensify")
    , m_centroidEmbs(centroidEmbs)
    , m_hp_isCached(maxNumWorkingDocs, "m_hp_isCached")
    , m_cachedWorkingIdxToDocIdx(maxNumWorkingDocs, kInvalidDocIdx)
{
    std::sort(residentPartitionConfigs.begin(), residentPartitionConfigs.end());

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

    for (const auto& residentPartitionConfig : residentPartitionConfigs)
    {
        m_residentEmbDatasets.push_back(ResidentEmbDataset(numDocs, residentPartitionConfig));
    }
}

void EmbDatasetManager::update(const std::vector<T_DOC_IDX>& docIdxList, const std::vector<std::vector<T_EMB>>& emb2D)
{
    if (docIdxList.size() > m_numDocs)
    {
        std::ostringstream oss;
        oss << "docIdxList.size() > m_numDocs: " << docIdxList.size() << " > " << m_numDocs;
        throw std::runtime_error(oss.str());
    }

    for (size_t residentPartitionIdx = 0; residentPartitionIdx < m_residentEmbDatasets.size(); ++residentPartitionIdx)
    {
        auto& embDataset = m_residentEmbDatasets[residentPartitionIdx];
        embDataset.update(docIdxList, emb2D);
    }

    // Compute nearest centroid for each doc (L2 distance over all dims)
    std::vector<int> centroidIdxList(docIdxList.size());
    size_t numCentroids = m_centroidEmbs.size();
    for (size_t i = 0; i < docIdxList.size(); ++i)
    {
        float bestDist = std::numeric_limits<float>::max();
        int bestCentroid = 0;
        for (size_t c = 0; c < numCentroids; ++c)
        {
            float dist = 0.0f;
            for (size_t d = 0; d < m_totalEmbDim; ++d)
            {
                float diff = static_cast<float>(emb2D[i][d]) - m_centroidEmbs[c][d];
                dist += diff * diff;
            }
            if (dist < bestDist)
            {
                bestDist = dist;
                bestCentroid = static_cast<int>(c);
            }
        }
        centroidIdxList[i] = bestCentroid;
    }

    m_resQuantDataset.update(docIdxList, emb2D, centroidIdxList);
}

const WorkingEmbDataset& EmbDatasetManager::densify(std::vector<T_DOC_IDX>& docIdxList,
                                                    size_t embIdxBeginIncl,
                                                    size_t embIdxEndExcl,
                                                    MemLayout memLayout)
{
    if (docIdxList.size() > m_maxNumWorkingDocs)
    {
        std::ostringstream oss;
        oss << "docIdxList.size() > m_maxNumWorkingDocs: " << docIdxList.size() << " > " << m_maxNumWorkingDocs;
        throw std::runtime_error(oss.str());
    }
    // ------------
    // Cache the docIdxList.
    cache(docIdxList);

    CHECK_CUDA(cudaMemcpy(m_docIdxListToDensify.data(),
                          docIdxList.data(),
                          docIdxList.size() * sizeof(T_DOC_IDX),
                          cudaMemcpyHostToDevice));

    m_workingEmbDataset.setMemLayout(memLayout);
    m_workingEmbDataset.setEmbDimBeginIncl(embIdxBeginIncl);
    m_workingEmbDataset.setEmbDimEndExcl(embIdxEndExcl);

    DensificationTask densificationTask;
    densificationTask.numDocsToDensify = docIdxList.size();
    densificationTask.globalEmbIdxBeginIncl = embIdxBeginIncl;
    densificationTask.globalEmbIdxEndExcl = embIdxEndExcl;
    densificationTask.d_workingEmbDataset = m_workingEmbDataset.data();
    densificationTask.d_docIdxList = m_docIdxListToDensify.data();
    densificationTask.hp_isCached = m_hp_isCached.data();

    for (size_t residentPartitionIdx = 0; residentPartitionIdx < m_residentEmbDatasets.size(); ++residentPartitionIdx)
    {
        const auto& embDataset = m_residentEmbDatasets[residentPartitionIdx];
        embDataset.densify(densificationTask);
    }

    m_resQuantDataset.densifyCompressed(densificationTask);

    return m_workingEmbDataset;
}

void EmbDatasetManager::cache(std::vector<T_DOC_IDX>& docIdxList)
{
    // ------------
    // Verify docIdxList is unique.
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
    std::queue<T_DOC_IDX> uncachedDocIdxList;
    int cnt1 = 0;
    int cnt2 = 0;
    for (T_DOC_IDX workingIdx = 0; workingIdx < docIdxList.size(); ++workingIdx)
    {
        T_DOC_IDX docIdx = docIdxList[workingIdx];
        auto it = m_cachedDocIdxToWorkingIdx.find(docIdx);
        if (it != m_cachedDocIdxToWorkingIdx.end() && it->second < docIdxList.size())
        {
            T_DOC_IDX cachedWorkingIdx = it->second;
            reorderedDocIdxList[cachedWorkingIdx] = docIdx;
            m_hp_isCached.data()[cachedWorkingIdx] = 1;
            cnt1++;
        }
        else
        {
            // Very important: if the cached working index is larger than the docIdxList.size(),
            //                 it is still considered as uncached.
            uncachedDocIdxList.push(docIdx);
            cnt2++;
        }
    }

    std::cout << "cnt1: " << cnt1 << ", cnt2: " << cnt2 << std::endl;

    // ------------
    // Verify no two docIdx map to the same cachedWorkingIdx.
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
    int cnt3 = 0;
    int cnt4 = 0;
    for (T_DOC_IDX workingIdx = 0; workingIdx < docIdxList.size(); ++workingIdx)
    {
        if (reorderedDocIdxList[workingIdx] == kInvalidDocIdx)
        {
            cnt3++;
            T_DOC_IDX uncachedDocIdx = uncachedDocIdxList.front();
            reorderedDocIdxList[workingIdx] = uncachedDocIdx;
            uncachedDocIdxList.pop();
            T_DOC_IDX oldDocIdx = m_cachedWorkingIdxToDocIdx[workingIdx];
            if (oldDocIdx != kInvalidDocIdx)
            {
                m_cachedDocIdxToWorkingIdx.erase(oldDocIdx);
            }
            m_cachedDocIdxToWorkingIdx[uncachedDocIdx] = workingIdx;
            m_cachedWorkingIdxToDocIdx[workingIdx] = uncachedDocIdx;
            m_hp_isCached.data()[workingIdx] = 0;
        }
        else
        {
            cnt4++;
        }
    }
    std::cout << "cnt1 cnt2 cnt3 cnt4: " << cnt1 << " " << cnt2 << " " << cnt3 << " " << cnt4 << std::endl;

    // ------------
    // Verify the uncachedDocIdxList is empty.
    if (!uncachedDocIdxList.empty())
    {
        std::ostringstream oss;
        oss << "Uncached doc indices are not empty: " << uncachedDocIdxList.size();
        throw std::runtime_error(oss.str());
    }

    // ------------
    // Reassign the reorderedDocIdxList to the docIdxList.
    docIdxList = reorderedDocIdxList;
}
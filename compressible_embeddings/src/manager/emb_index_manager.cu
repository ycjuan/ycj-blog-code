#include <algorithm>
#include <limits>
#include <queue>

#include "common/typedef.hpp"
#include "manager/emb_index_manager.hpp"
#include "utils/util.hpp"

EmbIndexManager::EmbIndexManager(size_t numDocs,
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
    , m_resQuantIndex(numDocs, totalEmbDim, maxNumWorkingDocs, m_compressedPartitionConfigs,
                      numBitsPerDim, centroidEmbs, centroidStdDevs)
    , m_workingEmbIndex(maxNumWorkingDocs, totalEmbDim)
    , m_docIdxListToDensify(maxNumWorkingDocs, "m_docIdxListToDensify")
    , m_centroidEmbs(centroidEmbs)
    , m_hp_isCached(maxNumWorkingDocs, "m_hp_isCached")
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
        m_residentEmbIndices.push_back(ResidentEmbIndex(numDocs, residentPartitionConfig));
    }
}

void EmbIndexManager::update(const std::vector<T_DOC_IDX>& docIdxList, const std::vector<std::vector<T_EMB>>& emb2D)
{
    if (docIdxList.size() > m_numDocs)
    {
        std::ostringstream oss;
        oss << "docIdxList.size() > m_numDocs: " << docIdxList.size() << " > " << m_numDocs;
        throw std::runtime_error(oss.str());
    }

    for (size_t residentPartitionIdx = 0; residentPartitionIdx < m_residentEmbIndices.size(); ++residentPartitionIdx)
    {
        auto& embIndex = m_residentEmbIndices[residentPartitionIdx];
        embIndex.update(docIdxList, emb2D);
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

    m_resQuantIndex.update(docIdxList, emb2D, centroidIdxList);
}

const WorkingEmbIndex& EmbIndexManager::densify(std::vector<T_DOC_IDX>& docIdxList, size_t embIdxBeginIncl, size_t embIdxEndExcl, MemLayout memLayout)
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

    m_workingEmbIndex.setMemLayout(memLayout);
    m_workingEmbIndex.setEmbDimBeginIncl(embIdxBeginIncl);
    m_workingEmbIndex.setEmbDimEndExcl(embIdxEndExcl);

    DensificationTask densificationTask;
    densificationTask.numDocsToDensify = docIdxList.size();
    densificationTask.globalEmbIdxBeginIncl = embIdxBeginIncl;
    densificationTask.globalEmbIdxEndExcl = embIdxEndExcl;
    densificationTask.d_workingEmbIndex = m_workingEmbIndex.data();
    densificationTask.d_docIdxList = m_docIdxListToDensify.data();
    densificationTask.hp_isCached = m_hp_isCached.data();

    for (size_t residentPartitionIdx = 0; residentPartitionIdx < m_residentEmbIndices.size(); ++residentPartitionIdx)
    {
        const auto& embIndex = m_residentEmbIndices[residentPartitionIdx];
        embIndex.densify(densificationTask);
    }

    m_resQuantIndex.densifyCompressed(densificationTask);

    return m_workingEmbIndex;
}

void EmbIndexManager::cache(std::vector<T_DOC_IDX>& docIdxList)
{
    constexpr T_DOC_IDX kInvalidDocIdx = -1;

    // ------------
    // First scan to find the cached working indices.
    std::vector<T_DOC_IDX> reorderedDocIdxList(docIdxList.size(), kInvalidDocIdx);
    std::queue<T_DOC_IDX> uncachedDocIdxList;
    for (T_DOC_IDX workingIdx = 0; workingIdx < docIdxList.size(); ++workingIdx)
    {
        T_DOC_IDX docIdx = docIdxList[workingIdx];
        auto it = m_cachedDocIdxToWorkingIdx.find(docIdx);
        if (it != m_cachedDocIdxToWorkingIdx.end() && it->second < docIdxList.size())
        {
            T_DOC_IDX cachedWorkingIdx = it->second;
            reorderedDocIdxList[cachedWorkingIdx] = docIdx;
            m_hp_isCached.data()[cachedWorkingIdx] = 1;
        }
        else 
        {
            // Very important: if the cached working index is larger than the docIdxList.size(),
            //                 it is still considered as uncached.
            uncachedDocIdxList.push(docIdx);
        }
    }

    // ------------
    // Second scan to put the uncached doc indices to the reorderedDocIdxList.
    for (T_DOC_IDX workingIdx = 0; workingIdx < docIdxList.size(); ++workingIdx)
    {
        if (reorderedDocIdxList[workingIdx] == kInvalidDocIdx)
        {
            if (uncachedDocIdxList.empty())
            {
                std::ostringstream oss;
                oss << "Uncached doc indices are empty. This should not happen.";
                throw std::runtime_error(oss.str());
            }
            T_DOC_IDX uncachedDocIdx = uncachedDocIdxList.front();
            reorderedDocIdxList[workingIdx] = uncachedDocIdx;
            uncachedDocIdxList.pop();
            m_cachedDocIdxToWorkingIdx[uncachedDocIdx] = workingIdx;
            m_hp_isCached.data()[workingIdx] = 0;
        }
    }

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
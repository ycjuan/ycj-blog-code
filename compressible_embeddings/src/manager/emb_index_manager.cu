#include <algorithm>

#include "common/typedef.hpp"
#include "manager/emb_index_manager.hpp"
#include "utils/util.hpp"

EmbIndexManager::EmbIndexManager(size_t numDocs,
                                 size_t totalEmbDim,
                                 std::vector<ResidentPartitionConfig> residentPartitionConfigs,
                                 size_t maxNumWorkingDocs)
    : m_numDocs(numDocs)
    , m_totalEmbDim(totalEmbDim)
    , m_maxNumWorkingDocs(maxNumWorkingDocs)
    , m_workingEmbIndex(maxNumWorkingDocs, totalEmbDim)
    , m_docIdxListToDensify(maxNumWorkingDocs, "m_docIdxListToDensify")
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
}

const WorkingEmbIndex& EmbIndexManager::densify(const std::vector<T_DOC_IDX>& docIdxList, size_t embIdxBeginIncl, size_t embIdxEndExcl, MemLayout memLayout)
{
    if (docIdxList.size() > m_maxNumWorkingDocs)
    {
        std::ostringstream oss;
        oss << "docIdxList.size() > m_maxNumWorkingDocs: " << docIdxList.size() << " > " << m_maxNumWorkingDocs;
        throw std::runtime_error(oss.str());
    }
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

    for (size_t residentPartitionIdx = 0; residentPartitionIdx < m_residentEmbIndices.size(); ++residentPartitionIdx)
    {
        const auto& embIndex = m_residentEmbIndices[residentPartitionIdx];
        embIndex.densify(densificationTask);
    }

    return m_workingEmbIndex;
}
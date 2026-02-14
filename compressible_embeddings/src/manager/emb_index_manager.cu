#include <algorithm>

#include "common/typedef.hpp"
#include "manager/emb_index_manager.hpp"
#include "utils/util.hpp"

EmbIndexManager::EmbIndexManager(size_t numDocs,
                                 size_t totalEmbDim,
                                 std::vector<ResidentPartitionConfig> residentPartitionConfigs,
                                 size_t maxWorkingSetSize)
    : m_numDocs(numDocs)
    , m_totalEmbDim(totalEmbDim)
    , m_maxWorkingSetSize(maxWorkingSetSize)
    , m_workingEmbIndex(maxWorkingSetSize, totalEmbDim)
    , m_docIdxListToCopy(maxWorkingSetSize, "m_docIdxListToCopy")
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
    if (docIdxList.size() > m_maxWorkingSetSize)
    {
        std::ostringstream oss;
        oss << "docIdxList.size() > m_maxWorkingSetSize: " << docIdxList.size() << " > " << m_maxWorkingSetSize;
        throw std::runtime_error(oss.str());
    }
    CHECK_CUDA(cudaMemcpy(m_docIdxListToCopy.data(),
                          docIdxList.data(),
                          docIdxList.size() * sizeof(T_DOC_IDX),
                          cudaMemcpyHostToDevice));

    m_workingEmbIndex.setMemLayout(memLayout);
    m_workingEmbIndex.setEmbDimBeginIncl(embIdxBeginIncl);
    m_workingEmbIndex.setEmbDimEndExcl(embIdxEndExcl);

    DensificationTask densificationTask;
    densificationTask.numTasks = docIdxList.size();
    densificationTask.embIdxBeginIncl = embIdxBeginIncl;
    densificationTask.embIdxEndExcl = embIdxEndExcl;
    densificationTask.d_workingSetEmbIndex = m_workingEmbIndex.data();
    densificationTask.d_docIdxMap = m_docIdxListToCopy.data();

    for (size_t residentPartitionIdx = 0; residentPartitionIdx < m_residentEmbIndices.size(); ++residentPartitionIdx)
    {
        const auto& embIndex = m_residentEmbIndices[residentPartitionIdx];
        embIndex.densify(densificationTask);
    }

    return m_workingEmbIndex;
}
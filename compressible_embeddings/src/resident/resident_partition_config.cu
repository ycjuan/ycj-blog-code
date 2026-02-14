#include "resident/resident_partition_config.hpp"

ResidentPartitionConfig::ResidentPartitionConfig(size_t embDimBeginIncl, size_t embDimEndExcl, MemLayout memLayout)
    : m_embDimBeginIncl(embDimBeginIncl)
    , m_embDimEndExcl(embDimEndExcl)
    , m_memLayout(memLayout)
{
}

bool ResidentPartitionConfig::operator<(const ResidentPartitionConfig& other) const
{
    if (m_embDimBeginIncl != other.m_embDimBeginIncl)
    {
        return m_embDimBeginIncl < other.m_embDimBeginIncl;
    }
    else
    {
        return m_embDimEndExcl < other.m_embDimEndExcl;
    }
}

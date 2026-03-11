#include <algorithm>

#include "compressible/compressible_partition_config.hpp"
#include "resident/resident_partition_config.hpp"

CompressiblePartitionConfig::CompressiblePartitionConfig(int embDimBeginIncl, int embDimEndExcl)
    : m_embDimBeginIncl(embDimBeginIncl)
    , m_embDimEndExcl(embDimEndExcl)
{
}

int CompressiblePartitionConfig::getEmbDimBeginIncl() const
{
    return m_embDimBeginIncl;
}

int CompressiblePartitionConfig::getEmbDimEndExcl() const
{
    return m_embDimEndExcl;
}

bool CompressiblePartitionConfig::operator<(const CompressiblePartitionConfig& other) const
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

// Any dims that are not covered by the resident partitions are compressible.
std::vector<CompressiblePartitionConfig> findCompressiblePartitionConfigs(std::vector<ResidentPartitionConfig> residentPartitionConfigs, int totalEmbDim)
{
    std::sort(residentPartitionConfigs.begin(), residentPartitionConfigs.end());
    std::vector<CompressiblePartitionConfig> compressiblePartitionConfigs;
    int currBeginIncl = 0;
    for (const auto& residentPartitionConfig : residentPartitionConfigs)
    {
        if (currBeginIncl < residentPartitionConfig.getEmbDimBeginIncl())
        {
            compressiblePartitionConfigs.push_back(
                CompressiblePartitionConfig(currBeginIncl, residentPartitionConfig.getEmbDimBeginIncl()));
        }
        currBeginIncl = residentPartitionConfig.getEmbDimEndExcl();
    }
    if (currBeginIncl < totalEmbDim)
    {
        compressiblePartitionConfigs.push_back(CompressiblePartitionConfig(currBeginIncl, totalEmbDim));
    }
    return compressiblePartitionConfigs;
}

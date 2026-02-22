#include <algorithm>

#include "compressible/compressible_partition_config.hpp"
#include "resident/resident_partition_config.hpp"

CompressiblePartitionConfig::CompressiblePartitionConfig(size_t embDimBeginIncl, size_t embDimEndExcl)
    : m_embDimBeginIncl(embDimBeginIncl)
    , m_embDimEndExcl(embDimEndExcl)
{
}

size_t CompressiblePartitionConfig::getEmbDimBeginIncl() const
{
    return m_embDimBeginIncl;
}

size_t CompressiblePartitionConfig::getEmbDimEndExcl() const
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

std::vector<CompressiblePartitionConfig> findCompressiblePartitionConfigs(std::vector<ResidentPartitionConfig> residentPartitionConfigs, size_t totalEmbDim)
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

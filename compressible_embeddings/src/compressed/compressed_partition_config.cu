#include <algorithm>

#include "compressed/compressed_partition_config.hpp"
#include "resident/resident_partition_config.hpp"

CompressedPartitionConfig::CompressedPartitionConfig(size_t embDimBeginIncl, size_t embDimEndExcl)
    : m_embDimBeginIncl(embDimBeginIncl)
    , m_embDimEndExcl(embDimEndExcl)
{
}

size_t CompressedPartitionConfig::getEmbDimBeginIncl() const
{
    return m_embDimBeginIncl;
}

size_t CompressedPartitionConfig::getEmbDimEndExcl() const
{
    return m_embDimEndExcl;
}

bool CompressedPartitionConfig::operator<(const CompressedPartitionConfig& other) const
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

std::vector<CompressedPartitionConfig> findCompressedPartitionConfigs(std::vector<ResidentPartitionConfig> residentPartitionConfigs, size_t totalEmbDim)
{
    std::sort(residentPartitionConfigs.begin(), residentPartitionConfigs.end());
    std::vector<CompressedPartitionConfig> compressedPartitionConfigs;
    int currBeginIncl = 0;
    for (const auto& residentPartitionConfig : residentPartitionConfigs)
    {
        if (currBeginIncl < residentPartitionConfig.getEmbDimBeginIncl())
        {
            compressedPartitionConfigs.push_back(
                CompressedPartitionConfig(currBeginIncl, residentPartitionConfig.getEmbDimBeginIncl()));
        }
        currBeginIncl = residentPartitionConfig.getEmbDimEndExcl();
    }
    if (currBeginIncl < totalEmbDim)
    {
        compressedPartitionConfigs.push_back(CompressedPartitionConfig(currBeginIncl, totalEmbDim));
    }
    return compressedPartitionConfigs;
}

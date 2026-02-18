#pragma once

#include <cstddef>
#include <vector>

class ResidentPartitionConfig;

class CompressedPartitionConfig
{
public:
    CompressedPartitionConfig(size_t embDimBeginIncl, size_t embDimEndExcl);

    size_t getEmbDimBeginIncl() const;
    size_t getEmbDimEndExcl() const;
    bool operator<(const CompressedPartitionConfig& other) const;

private:
    size_t m_embDimBeginIncl;
    size_t m_embDimEndExcl;
};

/*
For example, if we have embDim = 16, and residentPartitionConfigs = {{2, 5}, {4, 7}, {10, 13}},
then the compressedPartitionConfigs will be {{0, 2}, {7, 10}, {13, 16}}.
*/
std::vector<CompressedPartitionConfig> findCompressedPartitionConfigs(
    std::vector<ResidentPartitionConfig> residentPartitionConfigs,
    size_t globalEmbDim);

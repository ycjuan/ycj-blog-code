#pragma once

#include <vector>

class ResidentPartitionConfig;

class CompressiblePartitionConfig
{
public:
    CompressiblePartitionConfig(int embDimBeginIncl, int embDimEndExcl);

    int getEmbDimBeginIncl() const;
    int getEmbDimEndExcl() const;
    bool operator<(const CompressiblePartitionConfig& other) const;

private:
    int m_embDimBeginIncl;
    int m_embDimEndExcl;
};

/*
For example, if we have embDim = 16, and residentPartitionConfigs = {{2, 5}, {4, 7}, {10, 13}},
then the compressiblePartitionConfigs will be {{0, 2}, {7, 10}, {13, 16}}.
*/
std::vector<CompressiblePartitionConfig> findCompressiblePartitionConfigs(
    std::vector<ResidentPartitionConfig> residentPartitionConfigs,
    int globalEmbDim);

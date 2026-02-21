#pragma once

#include <cstddef>
#include <vector>

#include "resident/resident_partition_config.hpp"

class CompressedEmbDataset
{
public:
    CompressedEmbDataset(size_t numDocs, size_t globalEmbDim, std::vector<ResidentPartitionConfig> residentIndexConfigs, size_t maxNumDocsInWorkingIndex);
};
#pragma once

#include <vector>

#include "resident/resident_partition_config.hpp"
#include "common/typedef.hpp"

class CompressibleEmbDataset
{
public:
    CompressibleEmbDataset(T_DOC_IDX numDocs, int globalEmbDim, std::vector<ResidentPartitionConfig> residentIndexConfigs, int maxNumDocsInWorkingDataset);
};
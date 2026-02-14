#pragma once

#include <cuda_bf16.h>

#include "manager/emb_index_manager.hpp"

class CompressedEmbIndex
{
public:
    CompressedEmbIndex(size_t numDocs, size_t totalEmbDim, std::vector<ResidentPartitionConfig> residentIndexConfigs, size_t maxWorkingSetSize);
};
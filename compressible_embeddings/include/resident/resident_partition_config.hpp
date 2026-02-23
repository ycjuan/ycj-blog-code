#pragma once

#include "common/memory_layout.hpp"

class ResidentPartitionConfig
{
public:
    ResidentPartitionConfig() = default;
    ResidentPartitionConfig(int embDimBeginIncl, int embDimEndExcl, MemLayout memLayout);

    __device__ __host__ int getEmbDimBeginIncl() const { return m_embDimBeginIncl; }
    __device__ __host__ int getEmbDimEndExcl() const { return m_embDimEndExcl; }
    __device__ __host__ int getEmbDim() const { return m_embDimEndExcl - m_embDimBeginIncl; }
    bool operator<(const ResidentPartitionConfig& other) const;

private:
    int m_embDimBeginIncl = 0;
    int m_embDimEndExcl = 0;
    MemLayout m_memLayout = MemLayout::ROW_MAJOR;
};

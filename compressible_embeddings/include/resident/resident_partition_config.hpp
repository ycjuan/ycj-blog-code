#pragma once

#include <cstddef>

#include "common/memory_layout.hpp"

class ResidentPartitionConfig
{
public:
    ResidentPartitionConfig() = default;
    ResidentPartitionConfig(size_t embDimBeginIncl, size_t embDimEndExcl, MemLayout memLayout);

    __device__ __host__ size_t getEmbDimBeginIncl() const { return m_embDimBeginIncl; }
    __device__ __host__ size_t getEmbDimEndExcl() const { return m_embDimEndExcl; }
    __device__ __host__ size_t getEmbDim() const { return m_embDimEndExcl - m_embDimBeginIncl; }
    bool operator<(const ResidentPartitionConfig& other) const;

private:
    size_t m_embDimBeginIncl = 0;
    size_t m_embDimEndExcl = 0;
    MemLayout m_memLayout = MemLayout::ROW_MAJOR;
};

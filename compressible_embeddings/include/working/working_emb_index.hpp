#pragma once

#include <cuda_bf16.h>

#include "common/typedef.hpp"
#include "utils/cuda_malloc_raii.hpp"
#include "common/memory_layout.hpp"

class WorkingEmbDataset
{
public:
    WorkingEmbDataset(size_t maxNumDocs, size_t totalEmbDim);
    T_EMB* data() const;

    // Setters and getters
    void setMemLayout(MemLayout memLayout);
    MemLayout getMemLayout() const;

    void setEmbDimBeginIncl(size_t embDimBeginIncl);
    size_t getEmbDimBeginIncl() const;

    void setEmbDimEndExcl(size_t embDimEndExcl);
    size_t getEmbDimEndExcl() const;

private:
    size_t m_maxNumDocs;
    size_t m_totalEmbDim;
    MemLayout m_memLayout;
    size_t m_embDimBeginIncl;
    size_t m_embDimEndExcl;
    CudaDeviceArray<T_EMB> m_workingEmbIndex;
};

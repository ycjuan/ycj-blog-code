#pragma once

#include <cuda_bf16.h>

#include "common/typedef.hpp"
#include "utils/cuda_raii.hpp"
#include "common/memory_layout.hpp"

class WorkingEmbDataset
{
public:
    WorkingEmbDataset(int maxNumDocs, int totalEmbDim);
    T_EMB* data() const;

    // Setters and getters
    void setMemLayout(MemLayout memLayout);
    MemLayout getMemLayout() const;

    void setEmbDimBeginIncl(int embDimBeginIncl);
    int getEmbDimBeginIncl() const;

    void setEmbDimEndExcl(int embDimEndExcl);
    int getEmbDimEndExcl() const;

private:
    int m_maxNumDocs;
    int m_totalEmbDim;
    MemLayout m_memLayout;
    int m_embDimBeginIncl;
    int m_embDimEndExcl;
    CudaDeviceArray<T_EMB> m_workingEmbDataset;
};

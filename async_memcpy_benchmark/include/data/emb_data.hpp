#pragma once

#include "common/typedef.hpp"
#include "utils/cuda_raii.hpp"

#include <unordered_map>

class EmbData
{
public:
    EmbData(int maxNumDocs, int embDim);

    const T_EMB* data() const;

private:
    int m_maxNumDocs;
    int m_embDim;
    CudaDeviceArray<T_EMB> m_data;
    std::unordered_map<long, int> m_docId2Idx;
};

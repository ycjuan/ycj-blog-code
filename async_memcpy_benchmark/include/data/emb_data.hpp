#pragma once

#include "common/typedef.hpp"
#include "utils/cuda_raii.hpp"

#include <unordered_map>
#include <vector>

class EmbData
{
public:
    EmbData(int maxNumDocs, int embDim);

    const T_EMB* data() const;

    void update(const std::vector<long>& jobIds, const std::vector<std::vector<T_EMB>>& embData2D);

    std::vector<std::vector<long>> score(const std::vector<std::vector<T_EMB>>& reqEmb, int k) const;

private:
    int m_maxNumDocs;
    int m_embDim;
    CudaDeviceArray<T_EMB> m_data;
    std::unordered_map<long, int> m_docId2Idx;
};

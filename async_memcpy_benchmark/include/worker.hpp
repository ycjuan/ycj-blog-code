#pragma once

#include "common/typedef.hpp"
#include "utils/cuda_raii.hpp"

#include <unordered_map>
#include <vector>

class Worker
{
public:
    Worker(int maxNumDocs, int embDim);
    virtual ~Worker() = default;

    const T_EMB* data() const;

    virtual void updateEmbData(const std::vector<long>& jobIds, const std::vector<std::vector<T_EMB>>& embData2D) = 0;
    virtual void updateScalarData(const std::vector<long>& jobIds, const std::vector<float>& scalars) = 0;

    void score(const std::vector<std::vector<T_EMB>>& reqEmb) const;

protected:

    int m_maxNumDocs;
    int m_embDim;
    CudaDeviceArray<T_EMB> m_data;
    CudaDeviceArray<float> m_d_scalars;
    mutable CudaDeviceArray<float> m_d_scores;
    std::unordered_map<long, int> m_docId2Idx;
    std::vector<long> m_idxToDocId;
    CudaStream m_stream;
};



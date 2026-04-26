#pragma once

#include "common/typedef.hpp"
#include "utils/cuda_raii.hpp"

#include <tuple>
#include <unordered_map>
#include <vector>

class Worker
{
public:
    Worker(int maxNumDocs, int embDim);
    virtual ~Worker() = default;

    const T_EMB* data() const;

    virtual void update(const std::vector<long>& jobIds, const std::vector<std::vector<T_EMB>>& embData2D) = 0;

    virtual std::vector<std::vector<long>> score(const std::vector<std::vector<T_EMB>>& reqEmb, int k) const = 0;

protected:
    std::tuple<int, int, std::vector<int>> scoreCore(const std::vector<std::vector<T_EMB>>& reqEmb, int k) const;

    int m_maxNumDocs;
    int m_embDim;
    CudaDeviceArray<T_EMB> m_data;
    std::unordered_map<long, int> m_docId2Idx;
    std::vector<long> m_idxToDocId;
    CudaStream m_stream;
};



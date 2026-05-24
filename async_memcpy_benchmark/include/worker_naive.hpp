#pragma once

#include "worker.hpp"

#include <mutex>

class WorkerNaive : public Worker
{
public:
    WorkerNaive(int maxNumDocs, int embDim);

    void updateEmbData(const std::vector<long>& docIds, const std::vector<std::vector<T_EMB>>& embData2D) override;
    void updateScalarData(const std::vector<long>& docIds, const std::vector<float>& scalars) override;
    void score(const std::vector<T_EMB>& reqEmb, const std::vector<int>& targetRowIdxs) override;

private:
    std::mutex m_mutex;
};

#pragma once

#include "worker.hpp"

#include <mutex>

class WorkerNaive : public Worker
{
public:
    WorkerNaive(int maxNumDocs, int embDim);

    void upsertDoc(const std::vector<long>& docIds, const std::vector<std::vector<T_EMB>>& embData2D) override;
    void updateScalarData(const std::vector<long>& docIds, const std::vector<float>& scalars) override;
    void deleteDocs(const std::vector<long>& docIds) override;
    void score(const std::vector<T_EMB>& reqEmb, const std::vector<int>& targetRowIdxVec) override;

private:
    std::mutex m_mutex;
};

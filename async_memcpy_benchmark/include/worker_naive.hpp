#pragma once

#include "worker.hpp"

#include <mutex>

class WorkerNaive : public Worker
{
public:
    WorkerNaive(int maxNumDocs, int embDim);

    void upsertDocs(const std::vector<long>& v_docIds, const std::vector<std::vector<T_EMB>>& v_embData2D) override;
    void updateScalarData(const std::vector<long>& v_docIds, const std::vector<float>& v_scalars) override;
    void deleteDocs(const std::vector<long>& v_docIds) override;
    void score(const std::vector<T_EMB>& v_reqEmb, const std::vector<int>& v_targetRowIdxVec) override;

private:
    std::mutex m_mutex;
};

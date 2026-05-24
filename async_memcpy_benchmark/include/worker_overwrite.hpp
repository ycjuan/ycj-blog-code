#pragma once

#include "worker.hpp"

#include <mutex>

class WorkerOverwrite : public Worker
{
public:
    WorkerOverwrite(int maxNumDocs, int embDim);

    void upsertDocs(const std::vector<long>& v_docId, const std::vector<std::vector<T_EMB>>& v2_embData) override;
    void updateScalarData(const std::vector<long>& v_docId, const std::vector<float>& v_scalar) override;
    void deleteDocs(const std::vector<long>& v_docId) override;
    void score(const std::vector<T_EMB>& v_reqEmb, const std::vector<int>& v_targetRowIdx) override;

private:
    std::mutex m_readMutex; // locked only when dirty bits are modified
    std::mutex m_writeMutex; // locked when rowIdx<>docId map is modified
};

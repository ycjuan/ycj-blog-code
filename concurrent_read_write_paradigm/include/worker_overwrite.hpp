#pragma once

#include "worker.hpp"

#include <mutex>

class WorkerOverwrite : public Worker
{
public:
    WorkerOverwrite(int maxNumDocs, int embDim, int numScalars);

    void upsertDocs(const std::vector<long>& v_docId, const std::vector<std::vector<T_EMB>>& v2_embData) override;
    void updateScalarData(const std::vector<long>& v_docId, const std::vector<std::vector<float>>& v2_scalar) override;
    void deleteDocs(const std::vector<long>& v_docId) override;
    void score(const std::vector<T_EMB>& v_reqEmb,
               const std::vector<int>&   v_targetRowIdx,
               int                       targetScalarIdx) override;

private:
    std::mutex m_mapMutex;      // protects CPU docId<>rowIdx map
    std::mutex m_dirtyBitMutex; // protects GPU dirty-bit array (m_d_dirty)
};

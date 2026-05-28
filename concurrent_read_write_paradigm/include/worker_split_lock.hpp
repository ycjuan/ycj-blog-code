#pragma once

#include "worker.hpp"

#include <mutex>

// Two-mutex design without dirty bits:
//   m_writeMutex: protects CPU maps (docId<>rowIdx) during resolution
//   m_readMutex:  serializes GPU scatter kernels against the score kernel
//
// Unlike WorkerOverwrite, documents are never hidden (no dirty-bit fencing).
// Score blocks only during the scatter kernel itself, not during H2D or map resolution.
class WorkerSplitLock : public Worker
{
public:
    WorkerSplitLock(int maxNumDocs, int embDim, int numScalars);

    void upsertDocs(const std::vector<long>& v_docId, const std::vector<std::vector<T_EMB>>& v2_embData) override;
    void updateScalarData(const std::vector<long>& v_docId, const std::vector<std::vector<float>>& v2_scalar) override;
    void deleteDocs(const std::vector<long>& v_docId) override;
    void score(const std::vector<T_EMB>& v_reqEmb,
               const std::vector<int>&   v_targetRowIdx,
               int                       targetScalarIdx) override;

private:
    std::mutex m_mapMutex; // protects CPU docId<>rowIdx maps
    std::mutex m_gpuMutex; // serializes GPU scatter kernels vs score kernel
};

#pragma once

#include "worker.hpp"

#include <mutex>

// Four-mutex design — one mutex per protected resource:
//   m_mapMutex:      protects CPU docId<>rowIdx maps
//   m_embDataMutex:  serializes GPU emb scatter (upsert) vs score kernel
//   m_scalarMutex:   serializes GPU scalar scatter (updateScalar) vs score kernel
//   m_dirtyBitMutex: serializes GPU dirty-bit writes (delete) vs score kernel
//
// Writers hold m_mapMutex during map resolution, then the relevant GPU mutex
// during the scatter kernel. Score holds all three GPU mutexes simultaneously
// (acquired via std::scoped_lock to avoid deadlock). This allows upsert,
// updateScalar, and delete to run concurrently with each other, while each
// is still mutually exclusive with score on its specific GPU resource.
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
    std::mutex m_mapMutex;      // protects CPU docId<>rowIdx maps
    std::mutex m_embDataMutex;  // protects GPU embedding array (m_data)
    std::mutex m_scalarMutex;   // protects GPU scalar array (m_d_scalars)
    std::mutex m_dirtyBitMutex; // protects GPU dirty-bit array (m_d_dirty)
};

#pragma once

#include "worker.hpp"

#include <mutex>
#include <unordered_map>

// Base class for copy-on-write workers. Every upsert writes emb data to a fresh
// rowIdx and atomically flips dirty bits. Subclasses differ only in how scalars
// are handled: carryScalarsOnUpsert() is called between the emb scatter and the
// dirty-bit flip so subclasses can inject scalar GPU writes at the right moment.
class WorkerCopyOnWrite : public Worker
{
public:
    WorkerCopyOnWrite(int maxNumDocs, int embDim);

    void upsertDocs(const std::vector<long>& v_docId, const std::vector<std::vector<T_EMB>>& v2_embData) override;

    void deleteDocs(const std::vector<long>& v_docId) override;

protected:
    // Hook called after emb scatter, before dirty-bit flip.
    // Eager overrides this to carry scalars to the new rowIdx on GPU.
    // Lazy leaves this as a no-op — scalars are synced lazily in score().
    virtual void carryScalarsOnUpsert(const std::vector<long>& v_docId, const std::vector<int>& v_newRowIdx) { }

    std::mutex m_readMutex; // locked only when dirty bits are modified or read
    std::mutex m_writeMutex; // locked when rowIdx<>docId map or scalar map is modified

    std::unordered_map<long, float> m_docId2scalar; // CPU-side scalar store
};

// Eager variant: updateScalarData scatters scalars to GPU immediately and updates
// the CPU mirror. carryScalarsOnUpsert() carries the scalar to the new rowIdx.
// score() calls scoreImpl() directly — m_d_scalars is always up to date.
class WorkerCopyOnWriteEager : public WorkerCopyOnWrite
{
public:
    WorkerCopyOnWriteEager(int maxNumDocs, int embDim);

    void updateScalarData(const std::vector<long>& v_docId, const std::vector<float>& v_scalar) override;

    void score(const std::vector<T_EMB>& v_reqEmb, const std::vector<int>& v_targetRowIdx) override;

protected:
    void carryScalarsOnUpsert(const std::vector<long>& v_docId, const std::vector<int>& v_newRowIdx) override;
};

// Lazy variant: updateScalarData stores scalars on CPU only. upsertDocs does not
// touch scalars at all. score() syncs CPU scalars to GPU for target rows, then scores.
class WorkerCopyOnWriteLazy : public WorkerCopyOnWrite
{
public:
    WorkerCopyOnWriteLazy(int maxNumDocs, int embDim);

    void updateScalarData(const std::vector<long>& v_docId, const std::vector<float>& v_scalar) override;

    void score(const std::vector<T_EMB>& v_reqEmb, const std::vector<int>& v_targetRowIdx) override;
};

#pragma once

#include "worker.hpp"

#include <mutex>
#include <unordered_map>

// Base class for copy-on-write workers. Every upsert writes to a fresh rowIdx
// and atomically flips dirty bits, so concurrent readers never see partial writes.
// Scalars are always mirrored in m_docId2scalar (CPU) so upsertDocs can carry
// them to the new rowIdx regardless of which scalar variant is used.
class WorkerCopyOnWrite : public Worker
{
public:
    WorkerCopyOnWrite(int maxNumDocs, int embDim);

    void upsertDocs(const std::vector<long>& v_docId, const std::vector<std::vector<T_EMB>>& v2_embData) override;

    void deleteDocs(const std::vector<long>& v_docId) override;

protected:
    std::mutex m_readMutex; // locked only when dirty bits are modified or read
    std::mutex m_writeMutex; // locked when rowIdx<>docId map or scalar map is modified

    std::unordered_map<long, float> m_docId2scalar; // CPU-side scalar mirror
};

// Eager variant: updateScalarData scatters scalars to GPU immediately.
// score() calls scoreImpl() directly — m_d_scalars is always up to date.
class WorkerCopyOnWriteEager : public WorkerCopyOnWrite
{
public:
    WorkerCopyOnWriteEager(int maxNumDocs, int embDim);

    void updateScalarData(const std::vector<long>& v_docId, const std::vector<float>& v_scalar) override;

    void score(const std::vector<T_EMB>& v_reqEmb, const std::vector<int>& v_targetRowIdx) override;
};

// Lazy variant: updateScalarData stores scalars on CPU only.
// score() syncs CPU scalars to GPU for the target rows, then scores.
class WorkerCopyOnWriteLazy : public WorkerCopyOnWrite
{
public:
    WorkerCopyOnWriteLazy(int maxNumDocs, int embDim);

    void updateScalarData(const std::vector<long>& v_docId, const std::vector<float>& v_scalar) override;

    void score(const std::vector<T_EMB>& v_reqEmb, const std::vector<int>& v_targetRowIdx) override;
};

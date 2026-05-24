#pragma once

#include "worker.hpp"

#include <mutex>
#include <unordered_map>

class WorkerCOW : public Worker
{
public:
    WorkerCOW(int maxNumDocs, int embDim);

    // Always writes to a new rowIdx. Old rowIdx is marked dirty=1 and recycled.
    void upsertDocs(const std::vector<long>& v_docId, const std::vector<std::vector<T_EMB>>& v2_embData) override;

    // Stores scalar on CPU only. No GPU copy until score() is called.
    void updateScalarData(const std::vector<long>& v_docId, const std::vector<float>& v_scalar) override;

    void deleteDocs(const std::vector<long>& v_docId) override;

    // Syncs CPU scalars to GPU for target rows, then scores.
    void score(const std::vector<T_EMB>& v_reqEmb, const std::vector<int>& v_targetRowIdx) override;

private:
    std::mutex m_readMutex; // locked only when dirty bits are modified or read
    std::mutex m_writeMutex; // locked when rowIdx<>docId map or scalar map is modified

    std::unordered_map<long, float> m_docId2scalar; // CPU-side scalar store
};

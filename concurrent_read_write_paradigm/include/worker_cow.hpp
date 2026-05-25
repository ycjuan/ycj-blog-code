#pragma once

#include "worker.hpp"

#include <mutex>
#include <unordered_map>
#include <vector>

// All data produced by the COW resolve phase, consumed by the scatter and
// dirty-bit flip phases.
struct CopyOnWriteUpsertData
{
    std::vector<EmbElement> v_embElement;
    std::vector<int>        v_newRowIdx;      // new rowIdx per doc (parallel to input v_docId)
    std::vector<int>        v_oldDirtyRowIdx; // old rowIdxs to mark dirty (existing docs only)
};

// Base class for copy-on-write workers. Provides shared helpers for the emb
// scatter and dirty-bit flip phases. Subclasses own their full upsertDocs
// implementation so scalar logic only appears in the variant that needs it.
class WorkerCopyOnWrite : public Worker
{
public:
    WorkerCopyOnWrite(int maxNumDocs, int embDim, int numScalars);

    void deleteDocs(const std::vector<long>& v_docId) override;

protected:
    // Phase 1: resolve docId->rowIdx (COW), H2D emb data, scatter emb to m_data.
    // Fills d_newRowIdx (must be pre-allocated to v_docId.size()).
    // Returns empty upsertData if there is nothing to do.
    CopyOnWriteUpsertData resolveAndScatterEmb(const std::vector<long>&               v_docId,
                                               const std::vector<std::vector<T_EMB>>& v2_embData,
                                               CudaDeviceArray<int>&                  d_newRowIdx);

    // Phase 2: atomically flip dirty bits — hide old rowIdxs, reveal new rowIdxs.
    void commitDirtyBitFlip(const CopyOnWriteUpsertData& upsertData,
                            const CudaDeviceArray<int>&  d_newRowIdx,
                            int                          numDocs);

    std::mutex m_readMutex;  // locked only when dirty bits are modified or read
    std::mutex m_writeMutex; // locked when rowIdx<>docId map or scalar map is modified

    std::unordered_map<long, std::vector<float>> m_docId2scalar; // CPU-side scalar store
};

// Eager variant: updateScalarData scatters scalars to GPU immediately and updates
// the CPU mirror. upsertDocs carries the scalars to the new rowIdx between the emb
// scatter and the dirty-bit flip.
// score() calls scoreImpl() directly — m_d_scalars is always up to date.
class WorkerCopyOnWriteEager : public WorkerCopyOnWrite
{
public:
    WorkerCopyOnWriteEager(int maxNumDocs, int embDim, int numScalars);

    void upsertDocs(const std::vector<long>& v_docId, const std::vector<std::vector<T_EMB>>& v2_embData) override;

    void updateScalarData(const std::vector<long>& v_docId, const std::vector<std::vector<float>>& v2_scalar) override;

    void score(const std::vector<T_EMB>& v_reqEmb,
               const std::vector<int>&   v_targetRowIdx,
               int                       targetScalarIdx) override;

private:
    void carryScalarsToNewRowIdx(const std::vector<long>& v_docId, const std::vector<int>& v_newRowIdx);
};

// Lazy variant: updateScalarData stores scalars on CPU only. upsertDocs does not
// touch scalars at all. score() syncs CPU scalars to GPU for target rows, then scores.
class WorkerCopyOnWriteLazy : public WorkerCopyOnWrite
{
public:
    WorkerCopyOnWriteLazy(int maxNumDocs, int embDim, int numScalars);

    void upsertDocs(const std::vector<long>& v_docId, const std::vector<std::vector<T_EMB>>& v2_embData) override;

    void updateScalarData(const std::vector<long>& v_docId, const std::vector<std::vector<float>>& v2_scalar) override;

    void score(const std::vector<T_EMB>& v_reqEmb,
               const std::vector<int>&   v_targetRowIdx,
               int                       targetScalarIdx) override;
};

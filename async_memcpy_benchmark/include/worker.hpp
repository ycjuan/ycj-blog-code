#pragma once

#include "common/typedef.hpp"
#include "utils/cuda_raii.hpp"

#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

// One embedding dimension value paired with its destination row. Used to
// batch-scatter emb data to non-contiguous rows in a single kernel launch.
struct EmbElement
{
    int   rowIdx;
    T_EMB val;
};

// One scalar value for a specific (row, scalar index) pair. Used to
// scatter scalar updates without touching neighboring scalar columns.
struct ScalarElement
{
    int   rowIdx;
    int   scalarIdx;
    float val;
};

class Worker
{
public:
    Worker(int maxNumDocs, int embDim, int numScalars);
    virtual ~Worker() = default;

    const T_EMB* data() const;
    int          getHeadRowIdx() const { return m_headRowIdx; }

    virtual void upsertDocs(const std::vector<long>& v_docId, const std::vector<std::vector<T_EMB>>& v2_embData) = 0;
    virtual void updateScalarData(const std::vector<long>& v_docId, const std::vector<std::vector<float>>& v2_scalar)
        = 0;
    virtual void deleteDocs(const std::vector<long>& v_docId) = 0;

    // Caller is assumed to already know the rowIdxs to score, so no docId->rowIdx conversion is needed.
    virtual void score(const std::vector<T_EMB>& v_reqEmb, const std::vector<int>& v_targetRowIdx, int targetScalarIdx)
        = 0;

protected:
    void scoreImpl(const std::vector<T_EMB>& v_reqEmb, const std::vector<int>& v_targetRowIdx, int targetScalarIdx);

    // Resolves docIds to rowIdxs (inserting new docs into maps) and builds
    // a flat list of EmbElements. Must be called under the write mutex.
    std::vector<EmbElement> resolveAndBuildEmbElements(const std::vector<long>&               v_docId,
                                                       const std::vector<std::vector<T_EMB>>& v2_embData);

    // Removes docIds from maps and returns the freed rowIdxs.
    // Must be called under the write mutex.
    std::vector<int> resolveDeletedRowIdxs(const std::vector<long>& v_docId);

    // Resolves docIds to rowIdxs and builds a flat list of ScalarElements (one per doc per scalar).
    // Must be called under the write mutex.
    std::vector<ScalarElement> resolveScalarElements(const std::vector<long>&               v_docId,
                                                     const std::vector<std::vector<float>>& v2_scalar);

    int m_maxNumDocs; // capacity: total rows allocated in device arrays
    int m_embDim;
    int m_numScalars;
    int m_headRowIdx; // next fresh row to allocate when m_emptyRowIdxSet is empty

    // Flat device array of embeddings, row-major: m_data[rowIdx * m_embDim + dim]
    CudaDeviceArray<T_EMB> m_data;
    // Flat device array of scalars, row-major: m_d_scalars[rowIdx * m_numScalars + scalarIdx]
    CudaDeviceArray<float> m_d_scalars;
    // Per-target score output buffer, reused across scoreImpl calls.
    CudaDeviceArray<float> m_d_scores;
    // Per-row dirty bit (char). 1 = row is being written or has been deleted; scorers skip it.
    CudaDeviceArray<char> m_d_dirty;

    // CPU-side docId <-> rowIdx mapping. All access must be under the write mutex.
    std::unordered_map<long, int> m_docId2rowIdx;
    std::vector<long>             m_rowIdx2DocId;
    // Free list of rowIdxs released by deleteDocs. Drained before advancing m_headRowIdx.
    std::unordered_set<int> m_emptyRowIdxSet;

    // Separate streams so score H2D + kernel and write H2D + kernel can be issued independently.
    CudaStream m_readStream;
    CudaStream m_writeStream;
};

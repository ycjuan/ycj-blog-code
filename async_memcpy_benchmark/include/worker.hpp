#pragma once

#include "common/typedef.hpp"
#include "utils/cuda_raii.hpp"

#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

struct CopyElement
{
    int rowIdx;
    T_EMB val;
};

struct ScalarElement
{
    int rowIdx;
    float val;
};

class Worker
{
public:
    Worker(int maxNumDocs, int embDim);
    virtual ~Worker() = default;

    const T_EMB* data() const;

    virtual void upsertDocs(const std::vector<long>& v_docId, const std::vector<std::vector<T_EMB>>& v2_embData) = 0;
    virtual void updateScalarData(const std::vector<long>& v_docId, const std::vector<float>& v_scalar) = 0;
    virtual void deleteDocs(const std::vector<long>& v_docId) = 0;

    // Caller is assumed to already know the rowIdxs to score, so no docId->rowIdx conversion is needed.
    virtual void score(const std::vector<T_EMB>& v_reqEmb, const std::vector<int>& v_targetRowIdx) = 0;

protected:
    void scoreImpl(const std::vector<T_EMB>& v_reqEmb, const std::vector<int>& v_targetRowIdx);

    // Resolves docIds to rowIdxs (inserting new docs into maps) and builds
    // a flat list of CopyElements. Must be called under the write mutex.
    std::vector<CopyElement> resolveAndBuildCopyElements(const std::vector<long>& v_docId,
                                                         const std::vector<std::vector<T_EMB>>& v2_embData);

    // Removes docIds from maps and returns the freed rowIdxs.
    // Must be called under the write mutex.
    std::vector<int> resolveDeletedRowIdxs(const std::vector<long>& v_docId);

    // Resolves docIds to rowIdxs and builds a list of ScalarElements.
    // Must be called under the write mutex.
    std::vector<ScalarElement> resolveScalarElements(const std::vector<long>& v_docId,
                                                     const std::vector<float>& v_scalar);

    // meta data
    int m_maxNumDocs;
    int m_embDim;
    int m_headRowIdx;

    // cuda arrays
    CudaDeviceArray<T_EMB> m_data;
    CudaDeviceArray<float> m_d_scalars;
    CudaDeviceArray<float> m_d_scores;
    CudaDeviceArray<char> m_d_dirty;

    // docId <> rowIdx related
    std::unordered_map<long, int> m_docId2rowIdx;
    std::vector<long> m_rowIdx2DocId;
    std::unordered_set<int> m_emptyRowIdxSet;

    // cuda streams
    CudaStream m_readStream;
    CudaStream m_writeStream;
};

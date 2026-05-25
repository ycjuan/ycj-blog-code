#include "utils/util.hpp"
#include "worker.hpp"

#include <vector>

Worker::Worker(int maxNumDocs, int embDim, int numScalars)
    : m_maxNumDocs(maxNumDocs)
    , m_embDim(embDim)
    , m_numScalars(numScalars)
    , m_headRowIdx(0)
    , m_data(maxNumDocs * embDim, "Worker")
    , m_d_scalars(maxNumDocs * numScalars, "scalars")
    , m_d_scores(maxNumDocs, "scores")
    , m_d_dirty(maxNumDocs, "dirty")
    , m_rowIdx2DocId(maxNumDocs, -1)
{
    // Zero-initialize so un-upserted rows score 0 rather than garbage.
    CHECK_CUDA(cudaMemset(m_d_scalars.data(), 0, maxNumDocs * numScalars * sizeof(float)));
    // All rows start clean; dirty bits are set when a row is being written or deleted.
    CHECK_CUDA(cudaMemset(m_d_dirty.data(), 0, maxNumDocs * sizeof(DirtyBit)));
}

const T_EMB* Worker::data() const { return m_data.data(); }

std::vector<EmbElement> Worker::resolveAndBuildEmbElements(const std::vector<long>&               v_docId,
                                                           const std::vector<std::vector<T_EMB>>& v2_embData)
{
    std::vector<EmbElement> v_element;
    v_element.reserve(v_docId.size() * m_embDim);

    for (int i = 0; i < (int)v_docId.size(); i++)
    {
        auto it = m_docId2rowIdx.find(v_docId[i]);
        int  rowIdx;
        if (it == m_docId2rowIdx.end())
        {
            // New doc: recycle a freed row before bumping the head pointer.
            if (!m_emptyRowIdxSet.empty())
            {
                rowIdx = *m_emptyRowIdxSet.begin();
                m_emptyRowIdxSet.erase(m_emptyRowIdxSet.begin());
            }
            else
            {
                rowIdx = m_headRowIdx++;
            }
            m_docId2rowIdx[v_docId[i]] = rowIdx;
            m_rowIdx2DocId[rowIdx]     = v_docId[i];
        }
        else
        {
            // Existing doc: reuse the same rowIdx (in-place overwrite strategy).
            rowIdx = it->second;
        }
        for (int j = 0; j < m_embDim; j++)
        {
            v_element.push_back({ rowIdx, v2_embData[i][j] });
        }
    }

    return v_element;
}

std::vector<int> Worker::resolveDeletedRowIdxs(const std::vector<long>& v_docId)
{
    std::vector<int> v_deletedRowIdx;
    v_deletedRowIdx.reserve(v_docId.size());

    for (long docId : v_docId)
    {
        auto it = m_docId2rowIdx.find(docId);
        if (it == m_docId2rowIdx.end())
        {
            continue;
        }
        int rowIdx = it->second;
        m_docId2rowIdx.erase(it);
        m_rowIdx2DocId[rowIdx] = -1;
        m_emptyRowIdxSet.insert(rowIdx);
        v_deletedRowIdx.push_back(rowIdx);
    }

    return v_deletedRowIdx;
}

std::vector<ScalarElement> Worker::resolveScalarElements(const std::vector<long>&               v_docId,
                                                         const std::vector<std::vector<float>>& v2_scalar)
{
    std::vector<ScalarElement> v_scalarElement;
    v_scalarElement.reserve(v_docId.size() * m_numScalars);

    for (int i = 0; i < (int)v_docId.size(); i++)
    {
        auto it = m_docId2rowIdx.find(v_docId[i]);
        if (it == m_docId2rowIdx.end())
        {
            continue;
        }
        for (int j = 0; j < (int)v2_scalar[i].size(); j++)
        {
            v_scalarElement.push_back({ it->second, j, v2_scalar[i][j] });
        }
    }

    return v_scalarElement;
}

static __global__ void kn_score(float*          d_scores,
                                const T_EMB*    d_reqEmb,
                                const T_EMB*    d_docData,
                                const float*    d_scalars,
                                const int*      d_rowIdx,
                                const DirtyBit* d_dirty,
                                int             embDim,
                                int             numScalars,
                                int             targetScalarIdx,
                                int             numTargets)
{
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= numTargets)
    {
        return;
    }
    int r = d_rowIdx[t];
    // Skip rows that are being written or were deleted; score 0 so they rank last.
    if (d_dirty[r] == DirtyBit::DIRTY)
    {
        d_scores[t] = 0.0f;
        return;
    }
    const T_EMB* doc = d_docData + r * embDim;
    // Dot product in bfloat16 arithmetic; multiplied by a scalar filter weight.
    T_EMB dot = __float2bfloat16(0.0f);
    for (int i = 0; i < embDim; i++)
    {
        dot = __hadd(dot, __hmul(d_reqEmb[i], doc[i]));
    }
    d_scores[t] = __bfloat162float(dot) * d_scalars[r * numScalars + targetScalarIdx];
}

void Worker::scoreImpl(const std::vector<T_EMB>& v_reqEmb, const std::vector<int>& v_targetRowIdx, int targetScalarIdx)
{
    const int numTargets = v_targetRowIdx.size();
    if (numTargets == 0)
    {
        return;
    }

    CudaDeviceArray<T_EMB> d_reqEmb(m_embDim, "reqEmb");
    CHECK_CUDA(cudaMemcpyAsync(d_reqEmb.data(),
                               v_reqEmb.data(),
                               m_embDim * sizeof(T_EMB),
                               cudaMemcpyHostToDevice,
                               m_readStream.get()));

    CudaDeviceArray<int> d_rowIdx(numTargets, "rowIdx");
    CHECK_CUDA(cudaMemcpyAsync(d_rowIdx.data(),
                               v_targetRowIdx.data(),
                               numTargets * sizeof(int),
                               cudaMemcpyHostToDevice,
                               m_readStream.get()));

    // Grow score buffer on demand; shrinking is not needed since numTargets is fixed per bench run.
    if (m_d_scores.getArraySize() < (uint64_t)numTargets)
    {
        m_d_scores = CudaDeviceArray<float>(numTargets, "scores");
    }

    const int kBlockSize = 256;
    const int gridSize   = (numTargets + kBlockSize - 1) / kBlockSize;
    kn_score<<<gridSize, kBlockSize, 0, m_readStream.get()>>>(m_d_scores.data(),
                                                              d_reqEmb.data(),
                                                              m_data.data(),
                                                              m_d_scalars.data(),
                                                              d_rowIdx.data(),
                                                              m_d_dirty.data(),
                                                              m_embDim,
                                                              m_numScalars,
                                                              targetScalarIdx,
                                                              numTargets);
    CHECK_CUDA(cudaStreamSynchronize(m_readStream.get()));
}

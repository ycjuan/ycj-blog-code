#include "utils/util.hpp"
#include "worker.hpp"

#include <vector>

Worker::Worker(int maxNumDocs, int embDim)
    : m_maxNumDocs(maxNumDocs)
    , m_embDim(embDim)
    , m_headRowIdx(0)
    , m_data(maxNumDocs * embDim, "Worker")
    , m_d_scalars(maxNumDocs, "scalars")
    , m_d_scores(maxNumDocs, "scores")
    , m_d_dirty(maxNumDocs, "dirty")
    , m_rowIdx2DocId(maxNumDocs, -1)
{
    CHECK_CUDA(cudaMemset(m_d_scalars.data(), 0, maxNumDocs * sizeof(float)));
    CHECK_CUDA(cudaMemset(m_d_dirty.data(), 0, maxNumDocs * sizeof(char)));
}

const T_EMB* Worker::data() const { return m_data.data(); }

std::vector<CopyElement> Worker::resolveAndBuildCopyElements(const std::vector<long>& v_docId,
                                                             const std::vector<std::vector<T_EMB>>& v2_embData)
{
    std::vector<CopyElement> v_element;
    v_element.reserve(v_docId.size() * m_embDim);

    for (int i = 0; i < (int)v_docId.size(); i++)
    {
        auto it = m_docId2rowIdx.find(v_docId[i]);
        int rowIdx;
        if (it == m_docId2rowIdx.end())
        {
            // new doc: assign a rowIdx
            if (!m_emptyRowIdxSet.empty())
            {
                // reuse a freed slot
                rowIdx = *m_emptyRowIdxSet.begin();
                m_emptyRowIdxSet.erase(m_emptyRowIdxSet.begin());
            }
            else
            {
                // allocate next slot
                rowIdx = m_headRowIdx++;
            }
            // update maps
            m_docId2rowIdx[v_docId[i]] = rowIdx;
            m_rowIdx2DocId[rowIdx] = v_docId[i];
        }
        else
        {
            // existing doc: reuse its rowIdx
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

std::vector<ScalarElement> Worker::resolveScalarElements(const std::vector<long>& v_docId,
                                                         const std::vector<float>& v_scalar)
{
    std::vector<ScalarElement> v_scalarElement;
    v_scalarElement.reserve(v_docId.size());

    for (int i = 0; i < (int)v_docId.size(); i++)
    {
        auto it = m_docId2rowIdx.find(v_docId[i]);
        if (it == m_docId2rowIdx.end())
        {
            continue;
        }
        v_scalarElement.push_back({ it->second, v_scalar[i] });
    }

    return v_scalarElement;
}

__global__ void kn_score(float* d_scores,
                         const T_EMB* d_reqEmb,
                         const T_EMB* d_docData,
                         const float* d_scalars,
                         const int* d_rowIdx,
                         const char* d_dirty,
                         int embDim,
                         int numTargets)
{
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= numTargets)
    {
        return;
    }
    int r = d_rowIdx[t];
    if (d_dirty[r])
    {
        d_scores[t] = 0.0f;
        return;
    }
    const T_EMB* doc = d_docData + r * embDim;
    T_EMB dot = __float2bfloat16(0.0f);
    for (int i = 0; i < embDim; i++)
    {
        dot = __hadd(dot, __hmul(d_reqEmb[i], doc[i]));
    }
    d_scores[t] = __bfloat162float(dot) * d_scalars[r];
}

void Worker::scoreImpl(const std::vector<T_EMB>& v_reqEmb, const std::vector<int>& v_targetRowIdx)
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

    if (m_d_scores.getArraySize() < (uint64_t)numTargets)
    {
        m_d_scores = CudaDeviceArray<float>(numTargets, "scores");
    }

    const int kBlockSize = 256;
    const int gridSize = (numTargets + kBlockSize - 1) / kBlockSize;
    kn_score<<<gridSize, kBlockSize, 0, m_readStream.get()>>>(m_d_scores.data(),
                                                              d_reqEmb.data(),
                                                              m_data.data(),
                                                              m_d_scalars.data(),
                                                              d_rowIdx.data(),
                                                              m_d_dirty.data(),
                                                              m_embDim,
                                                              numTargets);
    CHECK_CUDA(cudaStreamSynchronize(m_readStream.get()));
}

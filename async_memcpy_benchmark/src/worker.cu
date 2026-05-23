#include "utils/util.hpp"
#include "worker.hpp"

#include <vector>

Worker::Worker(int maxNumDocs, int embDim)
    : m_maxNumDocs(maxNumDocs)
    , m_embDim(embDim)
    , m_data(maxNumDocs * embDim, "Worker")
    , m_d_scalars(maxNumDocs, "scalars")
    , m_d_scores(maxNumDocs, "scores")
    , m_idxToDocId(maxNumDocs, -1)
{
    CHECK_CUDA(cudaMemset(m_d_scalars.data(), 0, maxNumDocs * sizeof(float)));
}

const T_EMB* Worker::data() const { return m_data.data(); }

__global__ void scoreKernel(float* scores,
                            const T_EMB* reqEmb,
                            const T_EMB* docData,
                            const float* scalars,
                            const int* rowIds,
                            int embDim,
                            int numTargets)
{
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= numTargets)
    {
        return;
    }
    int rowId = rowIds[t];
    const T_EMB* doc = docData + rowId * embDim;
    float dot = 0.0f;
    for (int i = 0; i < embDim; i++)
    {
        dot += __bfloat162float(reqEmb[i]) * __bfloat162float(doc[i]);
    }
    scores[t] = dot * scalars[rowId];
}

void Worker::score(const std::vector<T_EMB>& reqEmb, const std::vector<int>& targetRowIds) const
{
    const int numTargets = targetRowIds.size();
    if (numTargets == 0)
    {
        return;
    }

    CudaDeviceArray<T_EMB> d_reqEmb(m_embDim, "reqEmb");
    CHECK_CUDA(cudaMemcpy(d_reqEmb.data(), reqEmb.data(), m_embDim * sizeof(T_EMB), cudaMemcpyHostToDevice));

    CudaDeviceArray<int> d_rowIds(numTargets, "rowIds");
    CHECK_CUDA(cudaMemcpy(d_rowIds.data(), targetRowIds.data(), numTargets * sizeof(int), cudaMemcpyHostToDevice));

    if (m_d_scores.getArraySize() < (uint64_t)numTargets)
    {
        m_d_scores = CudaDeviceArray<float>(numTargets, "scores");
    }

    const int kBlockSize = 256;
    const int gridSize = (numTargets + kBlockSize - 1) / kBlockSize;
    scoreKernel<<<gridSize, kBlockSize>>>(m_d_scores.data(),
                                          d_reqEmb.data(),
                                          m_data.data(),
                                          m_d_scalars.data(),
                                          d_rowIds.data(),
                                          m_embDim,
                                          numTargets);
}

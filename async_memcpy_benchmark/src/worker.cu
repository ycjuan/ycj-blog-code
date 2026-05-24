#include "utils/util.hpp"
#include "worker.hpp"

#include <vector>

Worker::Worker(int maxNumDocs, int embDim)
    : m_maxNumDocs(maxNumDocs)
    , m_embDim(embDim)
    , m_data(maxNumDocs * embDim, "Worker")
    , m_d_scalars(maxNumDocs, "scalars")
    , m_d_scores(maxNumDocs, "scores")
    , m_rowIdx2DocId(maxNumDocs, -1)
{
    CHECK_CUDA(cudaMemset(m_d_scalars.data(), 0, maxNumDocs * sizeof(float)));
}

const T_EMB* Worker::data() const { return m_data.data(); }

__global__ void scoreKernel(float* scores,
                            const T_EMB* reqEmb,
                            const T_EMB* docData,
                            const float* scalars,
                            const int* rowIdxs,
                            int embDim,
                            int numTargets)
{
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= numTargets)
    {
        return;
    }
    int rowIdx = rowIdxs[t];
    const T_EMB* doc = docData + rowIdx * embDim;
    T_EMB dot = __float2bfloat16(0.0f);
    for (int i = 0; i < embDim; i++)
    {
        dot = __hadd(dot, __hmul(reqEmb[i], doc[i]));
    }
    scores[t] = __bfloat162float(dot) * scalars[rowIdx];
}

void Worker::score(const std::vector<T_EMB>& reqEmb, const std::vector<int>& targetRowIdxs)
{
    const int numTargets = targetRowIdxs.size();
    if (numTargets == 0)
    {
        return;
    }

    CudaDeviceArray<T_EMB> d_reqEmb(m_embDim, "reqEmb");
    CHECK_CUDA(cudaMemcpyAsync(d_reqEmb.data(),
                               reqEmb.data(),
                               m_embDim * sizeof(T_EMB),
                               cudaMemcpyHostToDevice,
                               m_readStream.get()));

    CudaDeviceArray<int> d_rowIdx(numTargets, "rowIdx");
    CHECK_CUDA(cudaMemcpyAsync(d_rowIdx.data(),
                               targetRowIdxs.data(),
                               numTargets * sizeof(int),
                               cudaMemcpyHostToDevice,
                               m_readStream.get()));

    if (m_d_scores.getArraySize() < (uint64_t)numTargets)
    {
        m_d_scores = CudaDeviceArray<float>(numTargets, "scores");
    }

    const int kBlockSize = 256;
    const int gridSize = (numTargets + kBlockSize - 1) / kBlockSize;
    scoreKernel<<<gridSize, kBlockSize, 0, m_readStream.get()>>>(m_d_scores.data(),
                                                                 d_reqEmb.data(),
                                                                 m_data.data(),
                                                                 m_d_scalars.data(),
                                                                 d_rowIdx.data(),
                                                                 m_embDim,
                                                                 numTargets);
    CHECK_CUDA(cudaStreamSynchronize(m_readStream.get()));
}

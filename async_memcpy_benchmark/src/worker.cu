#include "worker.hpp"
#include "utils/util.hpp"

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

__global__ void scoreKernel(
    float* scores,
    const T_EMB* reqEmb,
    const T_EMB* docData,
    const float* scalars,
    const int* docIndices,
    int embDim,
    int numTargets)
{
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= numTargets)
    {
        return;
    }
    int docIdx = docIndices[t];
    const T_EMB* doc = docData + docIdx * embDim;
    float dot = 0.0f;
    for (int i = 0; i < embDim; i++)
    {
        dot += __bfloat162float(reqEmb[i]) * __bfloat162float(doc[i]);
    }
    scores[t] = dot * scalars[docIdx];
}

void Worker::score(const std::vector<T_EMB>& reqEmb, const std::vector<long>& targetJobIds) const
{
    const int numTargets = targetJobIds.size();

    // Resolve job IDs to doc indices, skip unknown IDs
    std::vector<int> docIndices;
    docIndices.reserve(numTargets);
    for (long jobId : targetJobIds)
    {
        auto it = m_docId2Idx.find(jobId);
        if (it != m_docId2Idx.end())
        {
            docIndices.push_back(it->second);
        }
    }
    if (docIndices.empty())
    {
        return;
    }
    const int numValid = docIndices.size();

    CudaDeviceArray<T_EMB> d_reqEmb(m_embDim, "reqEmb");
    CHECK_CUDA(cudaMemcpy(d_reqEmb.data(), reqEmb.data(), m_embDim * sizeof(T_EMB), cudaMemcpyHostToDevice));

    CudaDeviceArray<int> d_docIndices(numValid, "docIndices");
    CHECK_CUDA(cudaMemcpy(d_docIndices.data(), docIndices.data(), numValid * sizeof(int), cudaMemcpyHostToDevice));

    if (m_d_scores.getArraySize() < (uint64_t)numValid)
    {
        m_d_scores = CudaDeviceArray<float>(numValid, "scores");
    }

    const int kBlockSize = 256;
    const int gridSize = (numValid + kBlockSize - 1) / kBlockSize;
    scoreKernel<<<gridSize, kBlockSize>>>(
        m_d_scores.data(),
        d_reqEmb.data(),
        m_data.data(),
        m_d_scalars.data(),
        d_docIndices.data(),
        m_embDim,
        numValid);
}

#include <algorithm>
#include <random>
#include <thrust/execution_policy.h>
#include <thrust/sort.h>

#include "common.cuh"
#include "emb_data_gpu.cuh"

namespace
{

// Turns the raw (numQuery x numDocs) score matrix into a flat array of ScoredDoc so it can be
// sorted per-request in one shot.
__global__ void kn_buildScoredDoc(const EMB_T* d_scores, int numQuery, int numDocs, ScoredDoc* d_scoredDoc)
{
    int idx   = blockIdx.x * blockDim.x + threadIdx.x;
    int total = numQuery * numDocs;
    if (idx < total)
    {
        d_scoredDoc[idx].reqIdx = idx / numDocs;
        d_scoredDoc[idx].docId  = idx % numDocs;
        d_scoredDoc[idx].score  = (float)d_scores[idx];
    }
}

// After sorting, each request's top numDocs entries are contiguous but still spaced numDocs
// apart; this compacts the first numToReturn of each group into a dense buffer.
__global__ void kn_compactTopK(const ScoredDoc* d_scoredDocSorted,
                               int              numQuery,
                               int              numDocs,
                               int              numToReturn,
                               int*             d_topDocId,
                               float*           d_topScore)
{
    int idx   = blockIdx.x * blockDim.x + threadIdx.x;
    int total = numQuery * numToReturn;
    if (idx < total)
    {
        int reqIdx      = idx / numToReturn;
        int rank        = idx % numToReturn;
        int srcIdx      = reqIdx * numDocs + rank;
        d_topDocId[idx] = d_scoredDocSorted[srcIdx].docId;
        d_topScore[idx] = d_scoredDocSorted[srcIdx].score;
    }
}

} // namespace

void EmbDataGpu::init(int numDocs, int embDim, unsigned seed)
{
    numDocs_ = numDocs;
    embDim_  = embDim;

    CHECK_CUDA(cudaMalloc(&d_emb_, (size_t)numDocs_ * embDim_ * sizeof(EMB_T)));

    std::vector<EMB_T>                    h_emb((size_t)numDocs_ * embDim_);
    std::default_random_engine            generator(seed);
    std::uniform_real_distribution<float> distribution(-1.0f, 1.0f);
    for (auto& v : h_emb)
    {
        v = distribution(generator);
    }
    CHECK_CUDA(cudaMemcpy(d_emb_, h_emb.data(), h_emb.size() * sizeof(EMB_T), cudaMemcpyHostToDevice));

    CHECK_CUBLAS(cublasCreate(&handle_));
}

void EmbDataGpu::destroy()
{
    if (d_emb_)
    {
        CHECK_CUDA(cudaFree(d_emb_));
        d_emb_ = nullptr;
    }
    if (d_query_)
    {
        CHECK_CUDA(cudaFree(d_query_));
        d_query_ = nullptr;
    }
    if (d_scores_)
    {
        CHECK_CUDA(cudaFree(d_scores_));
        d_scores_ = nullptr;
    }
    if (d_scoredDoc_)
    {
        CHECK_CUDA(cudaFree(d_scoredDoc_));
        d_scoredDoc_ = nullptr;
    }
    if (handle_)
    {
        CHECK_CUBLAS(cublasDestroy(handle_));
        handle_ = nullptr;
    }
}

void EmbDataGpu::ensureCapacity(int numQuery)
{
    if ((size_t)numQuery > capQuery_)
    {
        if (d_query_)
        {
            CHECK_CUDA(cudaFree(d_query_));
        }
        if (d_scores_)
        {
            CHECK_CUDA(cudaFree(d_scores_));
        }
        capQuery_ = numQuery;
        CHECK_CUDA(cudaMalloc(&d_query_, capQuery_ * embDim_ * sizeof(EMB_T)));
        CHECK_CUDA(cudaMalloc(&d_scores_, capQuery_ * (size_t)numDocs_ * sizeof(EMB_T)));
    }
    size_t neededScoredDoc = (size_t)numQuery * numDocs_;
    if (neededScoredDoc > capScoredDoc_)
    {
        if (d_scoredDoc_)
        {
            CHECK_CUDA(cudaFree(d_scoredDoc_));
        }
        capScoredDoc_ = neededScoredDoc;
        CHECK_CUDA(cudaMalloc(&d_scoredDoc_, capScoredDoc_ * sizeof(ScoredDoc)));
    }
}

std::vector<RequestResult> EmbDataGpu::score(const std::vector<Query>& v_query, int numToReturn, cudaStream_t stream)
{
    int numQuery = (int)v_query.size();
    if (numQuery == 0)
    {
        return {};
    }
    numToReturn = std::min(numToReturn, numDocs_);
    ensureCapacity(numQuery);

    // ----------------
    // Copy the query batch to GPU (H2D). In a real system these would already live on GPU
    // (e.g. produced by an embedding model); we keep it simple here since this project is about
    // dispatch/partition/sharding overhead, not H2D transfer cost.
    std::vector<EMB_T> h_query((size_t)numQuery * embDim_);
    for (int i = 0; i < numQuery; i++)
    {
        std::copy(v_query[i].v_emb.begin(), v_query[i].v_emb.end(), h_query.begin() + (size_t)i * embDim_);
    }
    CHECK_CUDA(
        cudaMemcpyAsync(d_query_, h_query.data(), h_query.size() * sizeof(EMB_T), cudaMemcpyHostToDevice, stream));

    // ----------------
    // GEMM: scores(numQuery x numDocs) = query(numQuery x embDim) * emb(numDocs x embDim)^T
    CHECK_CUBLAS(cublasSetStream(handle_, stream));
    const float alpha = 1.0f;
    const float beta  = 0.0f;
    CHECK_CUBLAS(cublasSgemm(handle_,
                             CUBLAS_OP_T,
                             CUBLAS_OP_N,
                             numDocs_,
                             numQuery,
                             embDim_,
                             &alpha,
                             d_emb_,
                             embDim_,
                             d_query_,
                             embDim_,
                             &beta,
                             d_scores_,
                             numDocs_));

    // ----------------
    // Top-K per request: flatten to ScoredDoc, sort (grouped by request, descending score), then
    // compact the first numToReturn of each group.
    int kBlockSize     = 256;
    int totalScoredDoc = numQuery * numDocs_;
    kn_buildScoredDoc<<<(totalScoredDoc + kBlockSize - 1) / kBlockSize, kBlockSize, 0, stream>>>(d_scores_,
                                                                                                 numQuery,
                                                                                                 numDocs_,
                                                                                                 d_scoredDoc_);
    CHECK_CUDA(cudaGetLastError());

    thrust::sort(thrust::cuda::par.on(stream), d_scoredDoc_, d_scoredDoc_ + totalScoredDoc, ScoredDocComparator());

    std::vector<int>   h_topDocId(numQuery * numToReturn);
    std::vector<float> h_topScore(numQuery * numToReturn);
    int*               d_topDocId = nullptr;
    float*             d_topScore = nullptr;
    CHECK_CUDA(cudaMalloc(&d_topDocId, h_topDocId.size() * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_topScore, h_topScore.size() * sizeof(float)));

    int totalTopK = numQuery * numToReturn;
    kn_compactTopK<<<(totalTopK + kBlockSize - 1) / kBlockSize, kBlockSize, 0, stream>>>(d_scoredDoc_,
                                                                                         numQuery,
                                                                                         numDocs_,
                                                                                         numToReturn,
                                                                                         d_topDocId,
                                                                                         d_topScore);
    CHECK_CUDA(cudaGetLastError());

    CHECK_CUDA(cudaMemcpyAsync(h_topDocId.data(),
                               d_topDocId,
                               h_topDocId.size() * sizeof(int),
                               cudaMemcpyDeviceToHost,
                               stream));
    CHECK_CUDA(cudaMemcpyAsync(h_topScore.data(),
                               d_topScore,
                               h_topScore.size() * sizeof(float),
                               cudaMemcpyDeviceToHost,
                               stream));
    CHECK_CUDA(cudaStreamSynchronize(stream));

    CHECK_CUDA(cudaFree(d_topDocId));
    CHECK_CUDA(cudaFree(d_topScore));

    std::vector<RequestResult> v_result(numQuery);
    for (int i = 0; i < numQuery; i++)
    {
        v_result[i].reqId = v_query[i].reqId;
        v_result[i].v_docId.assign(h_topDocId.begin() + i * numToReturn, h_topDocId.begin() + (i + 1) * numToReturn);
        v_result[i].v_score.assign(h_topScore.begin() + i * numToReturn, h_topScore.begin() + (i + 1) * numToReturn);
    }
    return v_result;
}

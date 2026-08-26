#pragma once

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <vector>

#include "common.cuh"
#include "types.cuh"

// EmbDataGpu holds one shard's worth of document embeddings (numDocs x embDim) resident on GPU,
// and knows how to score a batch of query embeddings against them (GEMM) and reduce the result
// to a per-request top-K (batched sort). This is the smallest unit of "data + compute" reused by
// all three variants (A/B/C) in this experiment; what differs between variants is how many of
// these exist and how requests get routed to them.
class EmbDataGpu
{
public:
    void init(int numDocs, int embDim, unsigned seed);
    void destroy();

    // Scores v_query (batch of B requests) against this shard's numDocs_ documents and returns
    // the top numToReturn (docId, score) pairs per request. docId is local to this shard - the
    // caller is responsible for adding any partition/shard offset if a globally unique id is
    // needed.
    std::vector<RequestResult> score(const std::vector<Query>& v_query, int numToReturn, cudaStream_t stream);

    int numDocs() const { return numDocs_; }
    int embDim() const { return embDim_; }

private:
    int    numDocs_ = 0;
    int    embDim_  = 0;
    EMB_T* d_emb_   = nullptr; // numDocs_ x embDim_, row-major

    cublasHandle_t handle_ = nullptr;

    // Scratch device buffers, sized lazily to the largest batch seen so far to avoid
    // malloc/free churn on every request batch.
    EMB_T*     d_query_      = nullptr;
    EMB_T*     d_scores_     = nullptr; // numQuery x numDocs_, row-major
    ScoredDoc* d_scoredDoc_  = nullptr;
    size_t     capQuery_     = 0;
    size_t     capScoredDoc_ = 0;

    void ensureCapacity(int numQuery);
};

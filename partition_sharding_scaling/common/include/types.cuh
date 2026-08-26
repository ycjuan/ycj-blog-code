#pragma once

#include <algorithm>
#include <utility>
#include <vector>

// A single-request query fired by a caller. In the real system these arrive one-by-one (not
// batched) and get aggregated by a Dispatcher.
struct Query
{
    long long          reqId       = -1; // globally unique id, used to route results back to the caller
    int                partitionId = 0;  // e.g. country -> partition mapping, decided by the caller
    std::vector<float> v_emb;            // embDim
};

// Top-K result for one Query, scored against however many docs a given Retriever (or the
// unified EmbDataGpu in variant C) was responsible for.
struct RequestResult
{
    long long          reqId = -1;
    std::vector<int>   v_docId;
    std::vector<float> v_score;
};

// One (docId, score) candidate produced by GEMM, tagged with which row (request) it belongs to
// so a single batched sort can produce per-request top-K.
struct ScoredDoc
{
    int   reqIdx; // index of the request *within the current batch* (not Query::reqId)
    int   docId;  // local doc index within the shard being scored
    float score;
};

// Sorts by reqIdx ascending (to keep each request's candidates contiguous), then by score
// descending (so each request's top-K sit at the front of its contiguous block).
struct ScoredDocComparator
{
    __host__ __device__ bool operator()(const ScoredDoc& a, const ScoredDoc& b) const
    {
        if (a.reqIdx != b.reqIdx)
        {
            return a.reqIdx < b.reqIdx;
        }
        return a.score > b.score;
    }
};

// Merges several RequestResult's for the *same* logical request (e.g. one per shard) into a
// single top-K result.
inline RequestResult mergeTopK(const std::vector<RequestResult>& v_partial, int numToReturn)
{
    RequestResult merged;
    if (v_partial.empty())
    {
        return merged;
    }
    merged.reqId = v_partial[0].reqId;

    std::vector<std::pair<float, int>> v_candidate; // (score, docId)
    for (const auto& partial : v_partial)
    {
        for (size_t i = 0; i < partial.v_docId.size(); i++)
        {
            v_candidate.emplace_back(partial.v_score[i], partial.v_docId[i]);
        }
    }
    int numToKeep = std::min((int)v_candidate.size(), numToReturn);
    std::partial_sort(v_candidate.begin(),
                      v_candidate.begin() + numToKeep,
                      v_candidate.end(),
                      [](const auto& a, const auto& b) { return a.first > b.first; });

    merged.v_docId.resize(numToKeep);
    merged.v_score.resize(numToKeep);
    for (int i = 0; i < numToKeep; i++)
    {
        merged.v_score[i] = v_candidate[i].first;
        merged.v_docId[i] = v_candidate[i].second;
    }
    return merged;
}

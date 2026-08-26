#pragma once

#include <unordered_map>
#include <vector>

#include "retriever.cuh"
#include "thread_pool.cuh"
#include "types.cuh"

struct UnifiedConfig
{
    int numPartitions   = 1;
    int numShards       = 1;
    int numDocsPerShard = 100000;
    int embDim          = 64;
};

// Variant C ("hide partition/sharding under EmbDataGpu"): from the caller's point of view there
// is exactly one EmbDataGpu. Internally it still holds numPartitions x numShards physical
// shards, but grouping-by-partition, fanning out to shards, and merging results all happen
// inside score() - callers (and the single Dispatcher/Retriever above it) never need to know
// partitions or shards exist.
class UnifiedEmbDataGpu
{
public:
    void init(UnifiedConfig cfg, int poolSize);
    void destroy();

    // v_query may contain a mix of partitions; results come back in the same order as v_query.
    std::vector<RequestResult> score(const std::vector<Query>& v_query, int numToReturn);

private:
    UnifiedConfig                       cfg_;
    std::vector<std::vector<Retriever>> vv_retriever_; // [partition][shard], internal only
    ThreadPool                          pool_;
};

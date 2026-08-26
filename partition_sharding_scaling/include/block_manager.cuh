#pragma once

#include <chrono>
#include <future>
#include <memory>
#include <vector>

#include "dispatcher.cuh"
#include "retriever.cuh"
#include "types.cuh"

struct SystemConfig
{
    int                       numPartitions   = 1;
    int                       numShards       = 1;
    int                       numDocsPerShard = 100000;
    int                       embDim          = 64;
    int                       numToReturn     = 100;
    int                       batchSize       = 32;
    std::chrono::microseconds maxWait { 2000 };
};

// Variant A ("macro partition"): a Block is {Dispatcher, Retriever, EmbDataGpu}, and there are
// numPartitions x numShards of them. On submit(), BlockManager identifies the partition, fires
// one async call per shard directly to that shard's own Dispatcher, then merges the numShards
// partial top-K results. This is the baseline design: it does not scale well as numPartitions
// grows because the number of live Dispatcher threads (and their mutexes/condvars) grows
// linearly with numPartitions x numShards.
class BlockManager
{
public:
    void init(SystemConfig cfg);
    void destroy();

    std::future<RequestResult> submit(Query query);

private:
    SystemConfig                                          cfg_;
    std::vector<std::vector<Retriever>>                   v2_retriever_;  // [partition][shard]
    std::vector<std::vector<std::unique_ptr<Dispatcher>>> v2_dispatcher_; // [partition][shard],
                                                                          // unique_ptr because
                                                                          // Dispatcher's
                                                                          // mutex/condition_variable
                                                                          // make it non-movable
};

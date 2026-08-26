#pragma once

#include <chrono>
#include <future>
#include <memory>
#include <vector>

#include "dispatcher.cuh"
#include "retrieval_system.cuh"
#include "retriever.cuh"
#include "types.cuh"

// Variant A ("macro partition"): a Block is {Dispatcher, Retriever, EmbDataGpu}, and there are
// numPartitions x numShards of them. On submit(), BlockManager identifies the partition, fires
// one async call per shard directly to that shard's own Dispatcher, then merges the numShards
// partial top-K results. This is the baseline design: it does not scale well as numPartitions
// grows because the number of live Dispatcher threads (and their mutexes/condvars) grows
// linearly with numPartitions x numShards.
class BlockManager : public RetrievalSystem
{
public:
    void init(SystemConfig cfg) override;
    void destroy() override;

    std::future<RequestResult> submit(Query query) override;

private:
    SystemConfig                                          cfg_;
    std::vector<std::vector<Retriever>>                   vv_retriever_;  // [partition][shard]
    std::vector<std::vector<std::unique_ptr<Dispatcher>>> vv_dispatcher_; // [partition][shard],
                                                                          // unique_ptr because
                                                                          // Dispatcher's
                                                                          // mutex/condition_variable
                                                                          // make it non-movable
};

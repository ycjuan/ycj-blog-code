#pragma once

#include <condition_variable>
#include <deque>
#include <future>
#include <mutex>
#include <thread>
#include <vector>

#include "block_manager.cuh" // reuses SystemConfig
#include "retriever.cuh"
#include "thread_pool.cuh"
#include "types.cuh"

// Variant B ("dispatcher pulled out of the block"): there is exactly one Dispatcher instance
// (one background thread) for the whole system, responsible for batching requests per-partition.
// When a partition's batch is ready, instead of spinning up per-(partition,shard) OS threads
// (as variant A's Dispatcher-per-Block does), it hands the fan-out-to-shards-and-merge work to a
// small, fixed-size ThreadPool. The number of OS threads used is therefore independent of
// numPartitions x numShards - only the number of Retrievers (GPU-side state) still scales with
// it.
class SharedDispatcherSystem
{
public:
    void init(SystemConfig cfg, int poolSize);
    void destroy();

    std::future<RequestResult> submit(Query query);

private:
    struct PendingItem
    {
        Query                       query;
        std::promise<RequestResult> promise;
    };

    void run();
    void processBatch(int partitionId, std::vector<PendingItem> batch);

    SystemConfig                        cfg_;
    std::vector<std::vector<Retriever>> v2_retriever_; // [partition][shard]
    ThreadPool                          pool_;

    std::thread                          thread_;
    std::mutex                           mutex_;
    std::condition_variable              cv_;
    std::vector<std::deque<PendingItem>> v_queue_; // one queue per partition
    bool                                 stopFlag_ = false;
};

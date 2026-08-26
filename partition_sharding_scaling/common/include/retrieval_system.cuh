#pragma once

#include <chrono>
#include <future>

#include "types.cuh"

// Config shared by all three retrieval-system variants (method_a/b/c), so the same benchmark
// harness in test/main.cu can construct and drive any of them identically.
struct SystemConfig
{
    int                       numPartitions   = 1;
    int                       numShards       = 1;
    int                       numDocsPerShard = 100000;
    int                       embDim          = 64;
    int                       numToReturn     = 100;
    int                       batchSize       = 32;
    std::chrono::microseconds maxWait { 2000 };
    int                       threadPoolSize = 8; // used by method_b/method_c to bound fan-out threads
};

// Common interface implemented by all three variants being compared in this experiment:
//   - method_a_macro_partition::BlockManager
//   - method_b_shared_dispatcher::SharedDispatcherSystem
//   - method_c_unified::UnifiedSystem
// This lets test/main.cu drive all three through the same benchmark loop instead of
// duplicating it per variant.
class RetrievalSystem
{
public:
    virtual ~RetrievalSystem() = default;

    virtual void init(SystemConfig cfg) = 0;
    virtual void destroy()              = 0;

    // Submits a single request. Implementations are expected to internally batch concurrent
    // submissions (via a Dispatcher) before scoring them against the underlying doc data.
    virtual std::future<RequestResult> submit(Query query) = 0;
};

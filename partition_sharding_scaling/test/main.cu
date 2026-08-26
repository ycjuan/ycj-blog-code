// Benchmark harness for the "macro vs micro partition and sharding" blog post.
//
// We build the same logical retrieval system (Dispatcher -> [partition x shard] Retrievers ->
// EmbDataGpu) three different ways:
//   A. BlockManager       - Dispatcher lives inside each (partition, shard) Block (baseline)
//   B. SharedDispatcherSystem - one Dispatcher, still numPartitions x numShards Retrievers
//   C. UnifiedSystem      - one Dispatcher, one Retriever; partitions/shards hidden inside
//                           EmbDataGpu
//
// and sweep numPartitions to see how end-to-end latency/throughput degrades (or doesn't) for
// each design as the number of (partition, shard) units grows.

#include <atomic>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <thread>
#include <vector>

#include "block_manager.cuh"
#include "shared_dispatcher_system.cuh"
#include "unified_system.cuh"

namespace
{

struct LoadTestResult
{
    double throughputQps = 0;
    double avgLatencyMs  = 0;
    double p50LatencyMs  = 0;
    double p99LatencyMs  = 0;
};

// Drives `submitFn` continuously from numClientThreads for durationSec seconds, and reports
// throughput + latency percentiles. `submitFn` must be thread-safe.
template <typename SubmitFn>
LoadTestResult runLoadTest(SubmitFn submitFn, int numPartitions, int numClientThreads, double durationSec)
{
    std::atomic<bool>                stopFlag { false };
    std::atomic<long long>           totalCompleted { 0 };
    std::vector<std::vector<double>> v_threadLatencyMs(numClientThreads);

    auto clientLoop = [&](int threadIdx)
    {
        std::default_random_engine            generator(threadIdx);
        std::uniform_int_distribution<int>    partitionDist(0, numPartitions - 1);
        std::uniform_real_distribution<float> embDist(-1.0f, 1.0f);
        long long                             nextReqId = threadIdx * 1000000000LL;

        while (!stopFlag.load(std::memory_order_relaxed))
        {
            Query query;
            query.reqId       = nextReqId++;
            query.partitionId = partitionDist(generator);
            query.v_emb.resize(32);
            for (auto& v : query.v_emb)
            {
                v = embDist(generator);
            }

            auto start  = std::chrono::high_resolution_clock::now();
            auto future = submitFn(std::move(query));
            future.get();
            auto stop = std::chrono::high_resolution_clock::now();

            double latencyMs = std::chrono::duration_cast<std::chrono::microseconds>(stop - start).count() / 1000.0;
            v_threadLatencyMs[threadIdx].push_back(latencyMs);
            totalCompleted.fetch_add(1, std::memory_order_relaxed);
        }
    };

    std::vector<std::thread> v_client;
    v_client.reserve(numClientThreads);
    auto testStart = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < numClientThreads; i++)
    {
        v_client.emplace_back(clientLoop, i);
    }
    std::this_thread::sleep_for(std::chrono::duration<double>(durationSec));
    stopFlag.store(true, std::memory_order_relaxed);
    for (auto& t : v_client)
    {
        t.join();
    }
    auto   testStop    = std::chrono::high_resolution_clock::now();
    double wallTimeSec = std::chrono::duration<double>(testStop - testStart).count();

    std::vector<double> v_latencyMs;
    for (auto& v : v_threadLatencyMs)
    {
        v_latencyMs.insert(v_latencyMs.end(), v.begin(), v.end());
    }
    std::sort(v_latencyMs.begin(), v_latencyMs.end());

    LoadTestResult result;
    result.throughputQps = v_latencyMs.size() / wallTimeSec;
    if (!v_latencyMs.empty())
    {
        result.avgLatencyMs = std::accumulate(v_latencyMs.begin(), v_latencyMs.end(), 0.0) / v_latencyMs.size();
        result.p50LatencyMs = v_latencyMs[v_latencyMs.size() * 50 / 100];
        result.p99LatencyMs = v_latencyMs[std::min(v_latencyMs.size() - 1, v_latencyMs.size() * 99 / 100)];
    }
    return result;
}

void printHeader()
{
    std::cout << std::left << std::setw(14) << "variant" << std::setw(14) << "numPartitions" << std::setw(14)
              << "throughputQps" << std::setw(14) << "avgLatMs" << std::setw(14) << "p50LatMs" << std::setw(14)
              << "p99LatMs" << std::endl;
}

void printRow(const std::string& variant, int numPartitions, const LoadTestResult& r)
{
    std::cout << std::left << std::setw(14) << variant << std::setw(14) << numPartitions << std::setw(14) << std::fixed
              << std::setprecision(1) << r.throughputQps << std::setw(14) << r.avgLatencyMs << std::setw(14)
              << r.p50LatencyMs << std::setw(14) << r.p99LatencyMs << std::endl;
}

} // namespace

int main(int argc, char** argv)
{
    // Kept small by default so the demo runs quickly; bump these up on a real GPU box to see
    // the scaling behavior more clearly.
    const int    kNumShards        = 4;
    const int    kNumDocsPerShard  = 20000;
    const int    kEmbDim           = 32;
    const int    kNumToReturn      = 50;
    const int    kBatchSize        = 16;
    const auto   kMaxWait          = std::chrono::microseconds(2000);
    const int    kNumClientThreads = 8;
    const double kDurationSec      = 2.0;
    const int    kThreadPoolSize   = 8; // fixed, independent of numPartitions (variants B & C)

    std::vector<int> v_numPartitions = { 4, 16, 48, 96 };
    if (argc > 1)
    {
        v_numPartitions.clear();
        for (int i = 1; i < argc; i++)
        {
            v_numPartitions.push_back(std::atoi(argv[i]));
        }
    }

    printHeader();

    for (int numPartitions : v_numPartitions)
    {
        SystemConfig cfg;
        cfg.numPartitions   = numPartitions;
        cfg.numShards       = kNumShards;
        cfg.numDocsPerShard = kNumDocsPerShard;
        cfg.embDim          = kEmbDim;
        cfg.numToReturn     = kNumToReturn;
        cfg.batchSize       = kBatchSize;
        cfg.maxWait         = kMaxWait;

        // -------- Variant A: macro partition (Dispatcher inside each Block) --------
        {
            BlockManager system;
            system.init(cfg);
            auto result = runLoadTest([&](Query q) { return system.submit(std::move(q)); },
                                      numPartitions,
                                      kNumClientThreads,
                                      kDurationSec);
            printRow("A_macro", numPartitions, result);
            system.destroy();
        }

        // -------- Variant B: single shared Dispatcher, still per-shard Retrievers --------
        {
            SharedDispatcherSystem system;
            system.init(cfg, kThreadPoolSize);
            auto result = runLoadTest([&](Query q) { return system.submit(std::move(q)); },
                                      numPartitions,
                                      kNumClientThreads,
                                      kDurationSec);
            printRow("B_shared", numPartitions, result);
            system.destroy();
        }

        // -------- Variant C: partition/sharding hidden inside EmbDataGpu --------
        {
            UnifiedSystem system;
            system.init(cfg, kThreadPoolSize);
            auto result = runLoadTest([&](Query q) { return system.submit(std::move(q)); },
                                      numPartitions,
                                      kNumClientThreads,
                                      kDurationSec);
            printRow("C_unified", numPartitions, result);
            system.destroy();
        }
    }

    return 0;
}

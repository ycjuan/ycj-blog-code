#include "worker_cow.hpp"
#include "worker_naive.hpp"
#include "worker_overwrite.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <functional>
#include <iomanip>
#include <iostream>
#include <memory>
#include <mutex>
#include <random>
#include <string>
#include <thread>
#include <vector>

// ---- config ----

struct BenchConfig
{
    int maxNumDocs = 100000;
    int numRealDocs = 50000;
    int embDim = 128;
    int numScalars = 4;
    int numToScore = 1000;
    int updateBatchSize = 64;

    double scoreQps = 100.0;
    double upsertQps = 10.0; // each call = updateBatchSize docs
    double updateScalarQps = 20.0;

    double durationSec = 10.0;
};

// ---- timing helpers ----

using Clock = std::chrono::steady_clock;
using TimePoint = std::chrono::time_point<Clock>;

static double toMs(std::chrono::nanoseconds ns) { return ns.count() / 1e6; }

struct LatencyRecorder
{
    std::unique_ptr<std::mutex> mu = std::make_unique<std::mutex>();
    std::vector<double> v_latencyMs;

    void record(double ms)
    {
        std::lock_guard<std::mutex> lock(*mu);
        v_latencyMs.push_back(ms);
    }

    void report(const std::string& name) const
    {
        if (v_latencyMs.empty())
        {
            std::cout << "  " << name << ": no samples\n";
            return;
        }
        std::vector<double> sorted = v_latencyMs;
        std::sort(sorted.begin(), sorted.end());
        double sum = 0;
        for (double v : sorted)
            sum += v;
        double mean = sum / sorted.size();
        double p50 = sorted[sorted.size() * 50 / 100];
        double p99 = sorted[sorted.size() * 99 / 100];
        double realQps
            = sorted.size() / (sum / 1000.0); // calls / total_wall would be wrong; report count/duration separately
        std::cout << "  " << std::left << std::setw(16) << name << "  n=" << std::setw(6) << sorted.size()
                  << "  mean=" << std::setw(8) << std::fixed << std::setprecision(3) << mean << "ms"
                  << "  p50=" << std::setw(8) << p50 << "ms"
                  << "  p99=" << std::setw(8) << p99 << "ms\n";
    }

    int count() const { return (int)v_latencyMs.size(); }
};

// ---- data generators ----

static std::vector<T_EMB> randomEmb(int embDim, std::mt19937& rng)
{
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<T_EMB> v(embDim);
    for (auto& x : v)
        x = __float2bfloat16(dist(rng));
    return v;
}

static std::vector<std::vector<T_EMB>> randomEmbBatch(int n, int embDim, std::mt19937& rng)
{
    std::vector<std::vector<T_EMB>> v2(n);
    for (auto& v : v2)
        v = randomEmb(embDim, rng);
    return v2;
}

static std::vector<float> randomScalars(int numScalars, std::mt19937& rng)
{
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    std::vector<float> v(numScalars);
    for (auto& x : v)
        dist(rng);
    for (auto& x : v)
        x = dist(rng);
    return v;
}

// ---- bootstrap ----

static std::vector<long> bootstrap(Worker& worker, int numRealDocs, int embDim, int numScalars, std::mt19937& rng)
{
    std::vector<long> v_docId(numRealDocs);
    std::iota(v_docId.begin(), v_docId.end(), 1); // docIds 1..numRealDocs

    std::vector<std::vector<T_EMB>> v2_emb = randomEmbBatch(numRealDocs, embDim, rng);
    worker.upsertDocs(v_docId, v2_emb);

    std::vector<std::vector<float>> v2_scalar(numRealDocs);
    for (auto& v : v2_scalar)
        v = randomScalars(numScalars, rng);
    worker.updateScalarData(v_docId, v2_scalar);

    return v_docId;
}

// ---- rate-limited thread loop ----

// Calls fn() at the given QPS until stopFlag is set.
// Records latency of each call into recorder.
static void rateLimitedLoop(double qps,
                            std::function<void()> fn,
                            LatencyRecorder& recorder,
                            const std::atomic<bool>& stopFlag)
{
    using namespace std::chrono;
    auto intervalNs = nanoseconds((long long)(1e9 / qps));
    auto next = Clock::now();

    while (!stopFlag.load(std::memory_order_relaxed))
    {
        auto now = Clock::now();
        if (now < next)
        {
            std::this_thread::sleep_until(next);
        }
        next += intervalNs;

        auto t0 = Clock::now();
        fn();
        auto t1 = Clock::now();
        recorder.record(toMs(t1 - t0));
    }
}

// ---- benchmark one worker ----

struct BenchResult
{
    std::string workerName;
    LatencyRecorder scoreRec;
    LatencyRecorder upsertRec;
    LatencyRecorder updateScalarRec;
    double durationSec;

    void report() const
    {
        std::cout << "\n=== " << workerName << " ===\n";
        scoreRec.report("score");
        upsertRec.report("upsert");
        updateScalarRec.report("updateScalar");
        std::cout << "  observed QPS:"
                  << "  score=" << std::fixed << std::setprecision(1) << scoreRec.count() / durationSec
                  << "  upsert=" << upsertRec.count() / durationSec
                  << "  updateScalar=" << updateScalarRec.count() / durationSec << "\n";
    }
};

static void runBench(const std::string& name,
                     Worker& worker,
                     const BenchConfig& cfg,
                     const std::vector<long>& v_bootstrapDocId,
                     BenchResult& result)
{
    std::mt19937 rng(42);
    std::uniform_int_distribution<long> newDocIdDist(cfg.numRealDocs + 1, cfg.numRealDocs * 2);
    std::uniform_int_distribution<int> existingDocDist(0, (int)v_bootstrapDocId.size() - 1);

    result.workerName = name;
    result.durationSec = cfg.durationSec;

    std::atomic<bool> stopFlag(false);

    // score thread: random rowIdxs in [0, numRealDocs)
    std::thread scoreThread(
        [&]()
        {
            std::mt19937 threadRng(1);
            std::uniform_int_distribution<int> rowIdxDist(0, cfg.numRealDocs - 1);
            rateLimitedLoop(
                cfg.scoreQps,
                [&]()
                {
                    std::vector<int> v_targetRowIdx(cfg.numToScore);
                    for (auto& r : v_targetRowIdx)
                        r = rowIdxDist(threadRng);
                    std::vector<T_EMB> v_reqEmb = randomEmb(cfg.embDim, threadRng);
                    worker.score(v_reqEmb, v_targetRowIdx, 0);
                },
                result.scoreRec,
                stopFlag);
        });

    // upsert thread: half existing, half new docIds
    std::thread upsertThread(
        [&]()
        {
            std::mt19937 threadRng(2);
            long nextNewDocId = cfg.numRealDocs + 1;
            rateLimitedLoop(
                cfg.upsertQps,
                [&]()
                {
                    int n = cfg.updateBatchSize;
                    std::vector<long> v_docId(n);
                    for (int i = 0; i < n; i++)
                    {
                        if (i % 2 == 0)
                        {
                            // existing
                            v_docId[i] = v_bootstrapDocId[existingDocDist(threadRng)];
                        }
                        else
                        {
                            // new
                            v_docId[i] = nextNewDocId++;
                        }
                    }
                    std::vector<std::vector<T_EMB>> v2_emb = randomEmbBatch(n, cfg.embDim, threadRng);
                    worker.upsertDocs(v_docId, v2_emb);
                },
                result.upsertRec,
                stopFlag);
        });

    // updateScalar thread
    std::thread updateScalarThread(
        [&]()
        {
            std::mt19937 threadRng(3);
            rateLimitedLoop(
                cfg.updateScalarQps,
                [&]()
                {
                    int n = cfg.updateBatchSize;
                    std::vector<long> v_docId(n);
                    for (int i = 0; i < n; i++)
                        v_docId[i] = v_bootstrapDocId[existingDocDist(threadRng)];
                    std::vector<std::vector<float>> v2_scalar(n);
                    for (auto& v : v2_scalar)
                        v = randomScalars(cfg.numScalars, threadRng);
                    worker.updateScalarData(v_docId, v2_scalar);
                },
                result.updateScalarRec,
                stopFlag);
        });

    std::this_thread::sleep_for(std::chrono::duration<double>(cfg.durationSec));
    stopFlag.store(true, std::memory_order_relaxed);

    scoreThread.join();
    upsertThread.join();
    updateScalarThread.join();
}

// ---- main ----

int main()
{
    BenchConfig cfg;

    std::cout << "Config:\n"
              << "  maxNumDocs=" << cfg.maxNumDocs << "  numRealDocs=" << cfg.numRealDocs << "  embDim=" << cfg.embDim
              << "  numScalars=" << cfg.numScalars << "\n"
              << "  numToScore=" << cfg.numToScore << "  updateBatchSize=" << cfg.updateBatchSize << "\n"
              << "  scoreQps=" << cfg.scoreQps << "  upsertQps=" << cfg.upsertQps
              << "  updateScalarQps=" << cfg.updateScalarQps << "\n"
              << "  durationSec=" << cfg.durationSec << "\n";

    struct WorkerDef
    {
        std::string name;
        std::function<std::unique_ptr<Worker>()> factory;
    };

    std::vector<WorkerDef> v_workerDef = {
        { "WorkerNaive", [&]() { return std::make_unique<WorkerNaive>(cfg.maxNumDocs, cfg.embDim, cfg.numScalars); } },
        { "WorkerOverwrite",
          [&]() { return std::make_unique<WorkerOverwrite>(cfg.maxNumDocs, cfg.embDim, cfg.numScalars); } },
        { "WorkerCopyOnWriteEager",
          [&]() { return std::make_unique<WorkerCopyOnWriteEager>(cfg.maxNumDocs, cfg.embDim, cfg.numScalars); } },
        { "WorkerCopyOnWriteLazy",
          [&]() { return std::make_unique<WorkerCopyOnWriteLazy>(cfg.maxNumDocs, cfg.embDim, cfg.numScalars); } },
    };

    std::vector<BenchResult> v_result;
    v_result.reserve(v_workerDef.size()); // prevent reallocation — BenchResult is not copyable

    for (auto& def : v_workerDef)
    {
        std::cout << "\nBootstrapping " << def.name << "...\n";
        auto worker = def.factory();
        std::mt19937 rng(0);
        std::vector<long> v_docId = bootstrap(*worker, cfg.numRealDocs, cfg.embDim, cfg.numScalars, rng);

        std::cout << "Running benchmark...\n";
        v_result.emplace_back();
        runBench(def.name, *worker, cfg, v_docId, v_result.back());
    }

    std::cout << "\n\n========== RESULTS ==========\n";
    for (auto& r : v_result)
        r.report();

    return 0;
}

#include <chrono>
#include <iostream>
#include <future>
#include <deque>
#include <cassert>

#include "thread_pool.cuh"

long pseudoRandom(long x)
{
    x = (x * 1103515245 + 12345) & 0x7fffffff;
    x = (x >> 3) ^ (x << 7);
    x = (x * 16807) % 2147483647;
    return x;
}

long worker(int numRepeats, long seed)
{
    long x = seed;
    for (int i = 0; i < numRepeats; i++)
    {
        x = pseudoRandom(x);
    }
    return x;
}

struct ExpConfig
{
    int numRepeats;
    int maxConcurrency;
    int numReqs;

    void print()
    {
        std::cout << "\n\nnumRepeats: " << numRepeats << std::endl;
        std::cout << "maxConcurrency: " << maxConcurrency << std::endl;
        std::cout << "numReqs: " << numReqs << std::endl;
    }
};

std::vector<long> runExp(ExpConfig config, ThreadPool& threadPool, bool useThreadPool)
{
    std::cout << "Running experiment with " << (useThreadPool ? "thread pool" : "std::async") << std::endl;

    std::deque<std::future<long>> futures;
    std::vector<long> results;
    auto startTimePoint = std::chrono::high_resolution_clock::now();
    for (int reqIdx = 0; reqIdx < config.numReqs; reqIdx++)
    {
        // -------------
        // Dispatch the request
        if (useThreadPool)
        {
            futures.push_back(threadPool.enqueue([config, reqIdx]() {
                return worker(config.numRepeats, reqIdx);
            }));
        }
        else
        {
            futures.push_back(std::async(std::launch::async, [config, reqIdx]() {
                return worker(config.numRepeats, reqIdx);
            }));
        }

        // -------------
        // Wait for the request to complete
        while (!futures.empty())
        {
            auto &future = futures.front();
            if (future.wait_for(std::chrono::seconds(0)) == std::future_status::ready)
            {
                results.push_back(future.get());
                futures.pop_front();
            }
            else
            {
                break;
            }
        }
    }

    // -------------
    // Wait for all requests to complete
    for (auto &future : futures)
    {
        results.push_back(future.get());
    }

    // -------------
    // Print benchmark results
    auto endTimePoint = std::chrono::high_resolution_clock::now();
    long durationMicroSec = std::chrono::duration_cast<std::chrono::microseconds>(endTimePoint - startTimePoint).count();
    std::cout << "Duration: " << durationMicroSec << " microseconds" << std::endl;
    std::cout << "QPS: " << config.numReqs * 1000000.0 / durationMicroSec << std::endl;
    std::cout << "Avg latency: " << durationMicroSec * 1.0 / config.numReqs << " microseconds" << std::endl;

    return results;
}

int main()
{
    ExpConfig config;
    config.maxConcurrency = 4;
    ThreadPool threadPool(config.maxConcurrency);
    {
        config.numRepeats = 1000000;
        config.numReqs = 5000;
        config.print();
        auto result1 = runExp(config, threadPool, false);
        auto result2 = runExp(config, threadPool, true);
        assert(result1 == result2);
    }

    {
        config.numRepeats /= 10;
        config.numReqs *= 10;
        config.print();
        auto result1 = runExp(config, threadPool, false);
        auto result2 = runExp(config, threadPool, true);
        assert(result1 == result2);
    }

    {
        config.numRepeats /= 10;
        config.numReqs *= 10;
        config.print();
        auto result1 = runExp(config, threadPool, false);
        auto result2 = runExp(config, threadPool, true);
        assert(result1 == result2);
    }
    return 0;
}
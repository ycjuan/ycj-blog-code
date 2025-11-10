#include <chrono>
#include <iostream>
#include <future>
#include <vector>
#include <deque>
#include <thread>

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
    int targetQPS;
    int durationSec;

    void print()
    {
        std::cout << "numRepeats: " << numRepeats << std::endl;
        std::cout << "maxConcurrency: " << maxConcurrency << std::endl;
        std::cout << "targetQPS: " << targetQPS << std::endl;
    }
};

void runExp(ExpConfig config)
{
    config.print();

    int numReqs = config.targetQPS * config.durationSec;

    std::deque<std::future<long>> futures;
    auto startTimePoint = std::chrono::high_resolution_clock::now();
    for (int reqIdx = 0; reqIdx < numReqs; reqIdx++)
    {
        // -------------
        // Dispatch the request
        futures.push_back(std::async(std::launch::async, [config, reqIdx]() {
            return worker(config.numRepeats, reqIdx);
        }));

        // -------------
        // Wait for the request to complete
        while (!futures.empty() || futures.size() >= config.maxConcurrency)
        {
            auto &future = futures.front();
            if (future.wait_for(std::chrono::seconds(0)) == std::future_status::ready)
            {
                future.get();
                futures.pop_front();
            }
            else
            {
                if (futures.size() >= config.maxConcurrency)
                {
                    std::this_thread::sleep_for(std::chrono::microseconds(100));
                    continue;
                }
                else
                {
                    break;
                }
            }
        }

        auto currentTimePoint = std::chrono::high_resolution_clock::now();
        auto scheduledDispatchTimePoint = startTimePoint + std::chrono::microseconds(int64_t((int64_t)reqIdx * 1000000 / config.targetQPS));
        auto timeToWait = scheduledDispatchTimePoint - currentTimePoint;
        if (timeToWait > std::chrono::microseconds(0))
        {
            std::this_thread::sleep_for(timeToWait);
        }
        
    }

    for (auto &future : futures)
    {
        future.get();
    }

    auto endTimePoint = std::chrono::high_resolution_clock::now();
    double durationSec
        = std::chrono::duration_cast<std::chrono::microseconds>(endTimePoint - startTimePoint).count() / 1000000.0;
    std::cout << "Duration: " << durationSec << " seconds" << std::endl;
    std::cout << "QPS: " << numReqs / durationSec << std::endl;

}

void runExpWithThreadPool(ExpConfig config)
{
    int numReqs = config.targetQPS * config.durationSec;

    ThreadPool threadPool(config.maxConcurrency);
    std::vector<std::future<long>> futures;
    auto startTimePoint = std::chrono::high_resolution_clock::now();
    for (int reqIdx = 0; reqIdx < numReqs; reqIdx++)
    {
        futures.push_back(threadPool.enqueue([config, reqIdx]() {
            return worker(config.numRepeats, reqIdx);
        }));
    }
    for (auto &future : futures)
    {
        future.get();
    }

    auto endTimePoint = std::chrono::high_resolution_clock::now();
    double durationSec
        = std::chrono::duration_cast<std::chrono::microseconds>(endTimePoint - startTimePoint).count() / 1000000.0;
    std::cout << "Duration: " << durationSec << " seconds" << std::endl;
    std::cout << "QPS: " << numReqs / durationSec << std::endl;
}

int main()
{
    ExpConfig config;
    config.numRepeats = 100;
    config.maxConcurrency = 10;
    config.targetQPS = 100;
    config.durationSec = 10;
    runExp(config);
    runExpWithThreadPool(config);
    return 0;
}
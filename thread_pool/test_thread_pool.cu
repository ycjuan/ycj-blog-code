#include <chrono>
#include <iostream>
#include <future>
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
    int numReqs;

    void print()
    {
        std::cout << "\n\nnumRepeats: " << numRepeats << std::endl;
        std::cout << "maxConcurrency: " << maxConcurrency << std::endl;
        std::cout << "numReqs: " << numReqs << std::endl;
    }
};

void runExp(ExpConfig config)
{
    std::deque<std::future<long>> futures;
    auto startTimePoint = std::chrono::high_resolution_clock::now();
    for (int reqIdx = 0; reqIdx < config.numReqs; reqIdx++)
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

    }

    for (auto &future : futures)
    {
        future.get();
    }

    auto endTimePoint = std::chrono::high_resolution_clock::now();
    long durationMicroSec = std::chrono::duration_cast<std::chrono::microseconds>(endTimePoint - startTimePoint).count();
    std::cout << "Duration: " << durationMicroSec << " microseconds" << std::endl;
    std::cout << "QPS: " << config.numReqs * 1000000.0 / durationMicroSec << std::endl;
    std::cout << "Avg latency: " << durationMicroSec * 1.0 / config.numReqs << " microseconds" << std::endl;

}

void runExpWithThreadPool(ExpConfig config)
{
    ThreadPool threadPool(config.maxConcurrency);
    std::deque<std::future<long>> futures;
    auto startTimePoint = std::chrono::high_resolution_clock::now();
    for (int reqIdx = 0; reqIdx < config.numReqs; reqIdx++)
    {
        futures.push_back(threadPool.enqueue([config, reqIdx]() {
            return worker(config.numRepeats, reqIdx);
        }));


        // -------------
        // Wait for the request to complete
        while (!futures.empty())
        {
            auto &future = futures.front();
            if (future.wait_for(std::chrono::seconds(0)) == std::future_status::ready)
            {
                future.get();
                futures.pop_front();
            }
            else
            {
                break;
            }
        }

    }
    for (auto &future : futures)
    {
        future.get();
    }

    auto endTimePoint = std::chrono::high_resolution_clock::now();
    long durationMicroSec = std::chrono::duration_cast<std::chrono::microseconds>(endTimePoint - startTimePoint).count();
    std::cout << "Duration: " << durationMicroSec << " microseconds" << std::endl;
    std::cout << "QPS: " << config.numReqs * 1000000.0 / durationMicroSec << std::endl;
    std::cout << "Avg latency: " << durationMicroSec * 1.0 / config.numReqs << " microseconds" << std::endl;
}

int main()
{
    ExpConfig config;
    config.maxConcurrency = 4;
    {
        config.numRepeats = 1000000;
        config.numReqs = 5000;
        config.print();
        runExp(config);
        runExpWithThreadPool(config);
    }

    {
        config.numRepeats /= 10;
        config.numReqs *= 10;
        config.print();
        runExp(config);
        runExpWithThreadPool(config);
    }

    {
        config.numRepeats /= 10;
        config.numReqs *= 10;
        config.print();
        runExp(config);
        runExpWithThreadPool(config);
    }
    return 0;
}
#pragma once

#include <chrono>
#include <condition_variable>
#include <deque>
#include <functional>
#include <future>
#include <mutex>
#include <thread>

#include "types.cuh"

// Dispatcher aggregates single, independently-arriving Query submissions into batches (up to
// batchSize, or whatever has accumulated after maxWait elapses) and hands each batch to a
// caller-supplied BatchScorer. This models the "Dispatcher" described in the blog: callers fire
// requests one-by-one, and the Dispatcher is responsible for batching them before handing off to
// a Retriever (or, in variant B/C, to something that fans out to multiple Retrievers).
class Dispatcher
{
public:
    // Takes the accumulated batch of queries and must return one RequestResult per query, in
    // the same order.
    using BatchScorer = std::function<std::vector<RequestResult>(std::vector<Query>&)>;

    void start(int batchSize, std::chrono::microseconds maxWait, BatchScorer scorer);
    void stop();

    std::future<RequestResult> submit(Query query);

private:
    struct PendingItem
    {
        Query                       query;
        std::promise<RequestResult> promise;
    };

    void run();

    int                       batchSize_ = 1;
    std::chrono::microseconds maxWait_ { 0 };
    BatchScorer               scorer_;

    std::thread             thread_;
    std::mutex              mutex_;
    std::condition_variable cv_;
    std::deque<PendingItem> queue_;
    bool                    stopFlag_ = false;
};

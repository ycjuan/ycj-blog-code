#pragma once

#include <condition_variable>
#include <functional>
#include <future>
#include <mutex>
#include <queue>
#include <thread>
#include <vector>

// A small, fixed-size thread pool used by variant B/C to fan out work to multiple Retrievers
// without spawning one OS thread per (partition, shard) - which is exactly the scalability
// problem variant A suffers from as numPartitions grows.
class ThreadPool
{
public:
    void init(int numThreads)
    {
        for (int i = 0; i < numThreads; i++)
        {
            v_worker_.emplace_back(&ThreadPool::workerLoop, this);
        }
    }

    void destroy()
    {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            stopFlag_ = true;
        }
        cv_.notify_all();
        for (auto& worker : v_worker_)
        {
            if (worker.joinable())
            {
                worker.join();
            }
        }
        v_worker_.clear();
    }

    std::future<void> enqueue(std::function<void()> task)
    {
        auto              packagedTask = std::make_shared<std::packaged_task<void()>>(std::move(task));
        std::future<void> future       = packagedTask->get_future();
        {
            std::lock_guard<std::mutex> lock(mutex_);
            // Only the shared_ptr is captured here, so this closure stays CopyConstructible
            // (as required by std::function) even if `task` itself wraps move-only state.
            queue_.emplace([packagedTask] { (*packagedTask)(); });
        }
        cv_.notify_one();
        return future;
    }

    // Overload for move-only callables (e.g. a lambda that captures a std::vector of
    // move-only items). std::function<void()> requires its target to be CopyConstructible, so
    // such callables can't go through the overload above; instead we wrap them in a
    // packaged_task directly via a template, which only requires the callable to be invocable.
    template <typename F>
    std::future<void> enqueueMoveOnly(F&& f)
    {
        auto              packagedTask = std::make_shared<std::packaged_task<void()>>(std::forward<F>(f));
        std::future<void> future       = packagedTask->get_future();
        {
            std::lock_guard<std::mutex> lock(mutex_);
            queue_.emplace([packagedTask] { (*packagedTask)(); });
        }
        cv_.notify_one();
        return future;
    }

private:
    void workerLoop()
    {
        while (true)
        {
            std::function<void()> task;
            {
                std::unique_lock<std::mutex> lock(mutex_);
                cv_.wait(lock, [this] { return stopFlag_ || !queue_.empty(); });
                if (queue_.empty())
                {
                    if (stopFlag_)
                    {
                        return;
                    }
                    continue;
                }
                task = std::move(queue_.front());
                queue_.pop();
            }
            task();
        }
    }

    std::vector<std::thread>          v_worker_;
    std::mutex                        mutex_;
    std::condition_variable           cv_;
    std::queue<std::function<void()>> queue_;
    bool                              stopFlag_ = false;
};

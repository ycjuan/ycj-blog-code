#include "dispatcher.cuh"

void Dispatcher::start(int batchSize, std::chrono::microseconds maxWait, BatchScorer scorer)
{
    batchSize_ = batchSize;
    maxWait_   = maxWait;
    scorer_    = std::move(scorer);
    stopFlag_  = false;
    thread_    = std::thread(&Dispatcher::run, this);
}

void Dispatcher::stop()
{
    {
        std::lock_guard<std::mutex> lock(mutex_);
        stopFlag_ = true;
    }
    cv_.notify_all();
    if (thread_.joinable())
    {
        thread_.join();
    }
}

std::future<RequestResult> Dispatcher::submit(Query query)
{
    std::future<RequestResult> future;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        queue_.emplace_back();
        queue_.back().query = std::move(query);
        future              = queue_.back().promise.get_future();
    }
    cv_.notify_one();
    return future;
}

void Dispatcher::run()
{
    while (true)
    {
        std::vector<PendingItem> batch;
        {
            std::unique_lock<std::mutex> lock(mutex_);
            // Wait until we have a full batch, or maxWait_ has elapsed (in which case we take
            // whatever has accumulated so far), or we've been asked to stop.
            cv_.wait_for(lock, maxWait_, [this] { return stopFlag_ || (int)queue_.size() >= batchSize_; });

            if (queue_.empty())
            {
                if (stopFlag_)
                {
                    return;
                }
                continue;
            }

            int numToTake = std::min((int)queue_.size(), batchSize_);
            batch.reserve(numToTake);
            for (int i = 0; i < numToTake; i++)
            {
                batch.push_back(std::move(queue_.front()));
                queue_.pop_front();
            }
        }

        std::vector<Query> v_query;
        v_query.reserve(batch.size());
        for (auto& item : batch)
        {
            v_query.push_back(item.query);
        }

        std::vector<RequestResult> v_result = scorer_(v_query);

        for (size_t i = 0; i < batch.size(); i++)
        {
            batch[i].promise.set_value(v_result[i]);
        }
    }
}

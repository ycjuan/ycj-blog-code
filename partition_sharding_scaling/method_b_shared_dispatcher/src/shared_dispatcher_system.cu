#include "shared_dispatcher_system.cuh"

void SharedDispatcherSystem::init(SystemConfig cfg)
{
    cfg_ = cfg;
    vv_retriever_.resize(cfg_.numPartitions);
    v_queue_.resize(cfg_.numPartitions);
    for (int p = 0; p < cfg_.numPartitions; p++)
    {
        vv_retriever_[p].resize(cfg_.numShards);
        for (int s = 0; s < cfg_.numShards; s++)
        {
            unsigned seed = (unsigned)(p * cfg_.numShards + s);
            vv_retriever_[p][s].init(cfg_.numDocsPerShard, cfg_.embDim, seed);
        }
    }

    pool_.init(cfg_.threadPoolSize);
    stopFlag_ = false;
    thread_   = std::thread(&SharedDispatcherSystem::run, this);
}

void SharedDispatcherSystem::destroy()
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
    pool_.destroy();
    for (auto& v_retriever : vv_retriever_)
    {
        for (auto& retriever : v_retriever)
        {
            retriever.destroy();
        }
    }
}

std::future<RequestResult> SharedDispatcherSystem::submit(Query query)
{
    int                        partitionId = query.partitionId % cfg_.numPartitions;
    std::future<RequestResult> future;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        auto&                       queue = v_queue_[partitionId];
        queue.emplace_back();
        queue.back().query = std::move(query);
        future             = queue.back().promise.get_future();
    }
    cv_.notify_one();
    return future;
}

void SharedDispatcherSystem::run()
{
    while (true)
    {
        std::vector<std::pair<int, std::vector<PendingItem>>> v_readyBatch;
        {
            std::unique_lock<std::mutex> lock(mutex_);
            cv_.wait_for(lock,
                         cfg_.maxWait,
                         [this]
                         {
                             if (stopFlag_)
                             {
                                 return true;
                             }
                             for (auto& queue : v_queue_)
                             {
                                 if ((int)queue.size() >= cfg_.batchSize)
                                 {
                                     return true;
                                 }
                             }
                             return false;
                         });

            for (int p = 0; p < cfg_.numPartitions; p++)
            {
                auto& queue = v_queue_[p];
                if (queue.empty())
                {
                    continue;
                }
                int                      numToTake = std::min((int)queue.size(), cfg_.batchSize);
                std::vector<PendingItem> batch;
                batch.reserve(numToTake);
                for (int i = 0; i < numToTake; i++)
                {
                    batch.push_back(std::move(queue.front()));
                    queue.pop_front();
                }
                v_readyBatch.emplace_back(p, std::move(batch));
            }

            if (v_readyBatch.empty() && stopFlag_)
            {
                return;
            }
        }

        for (auto& partitionBatch : v_readyBatch)
        {
            pool_.enqueueMoveOnly(
                [this, partitionId = partitionBatch.first, batch = std::move(partitionBatch.second)]() mutable
                { processBatch(partitionId, std::move(batch)); });
        }
    }
}

void SharedDispatcherSystem::processBatch(int partitionId, std::vector<PendingItem> batch)
{
    std::vector<Query> v_query;
    v_query.reserve(batch.size());
    for (auto& item : batch)
    {
        v_query.push_back(item.query);
    }

    // Score against every shard of this partition (sequentially, within this one pooled task),
    // then merge per-query across shards.
    std::vector<std::vector<RequestResult>> v_shardResult(cfg_.numShards);
    for (int s = 0; s < cfg_.numShards; s++)
    {
        v_shardResult[s] = vv_retriever_[partitionId][s].score(v_query, cfg_.numToReturn);
    }

    for (size_t i = 0; i < batch.size(); i++)
    {
        std::vector<RequestResult> v_partial;
        v_partial.reserve(cfg_.numShards);
        for (int s = 0; s < cfg_.numShards; s++)
        {
            v_partial.push_back(v_shardResult[s][i]);
        }
        batch[i].promise.set_value(mergeTopK(v_partial, cfg_.numToReturn));
    }
}

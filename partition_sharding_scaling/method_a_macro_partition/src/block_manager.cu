#include <future>

#include "block_manager.cuh"

void BlockManager::init(SystemConfig cfg)
{
    cfg_ = cfg;
    v2_retriever_.resize(cfg_.numPartitions);
    v2_dispatcher_.resize(cfg_.numPartitions);

    for (int p = 0; p < cfg_.numPartitions; p++)
    {
        // std::mutex/condition_variable inside Dispatcher make it non-movable, so we must
        // reserve capacity up-front and only ever emplace_back (never resize/reallocate).
        v2_retriever_[p].reserve(cfg_.numShards);
        v2_dispatcher_[p].reserve(cfg_.numShards);
        for (int s = 0; s < cfg_.numShards; s++)
        {
            unsigned seed = (unsigned)(p * cfg_.numShards + s);
            v2_retriever_[p].emplace_back();
            v2_retriever_[p][s].init(cfg_.numDocsPerShard, cfg_.embDim, seed);

            Retriever* retriever   = &v2_retriever_[p][s];
            int        numToReturn = cfg_.numToReturn;
            v2_dispatcher_[p].push_back(std::make_unique<Dispatcher>());
            v2_dispatcher_[p][s]->start(cfg_.batchSize,
                                        cfg_.maxWait,
                                        [retriever, numToReturn](std::vector<Query>& v_query)
                                        { return retriever->score(v_query, numToReturn); });
        }
    }
}

void BlockManager::destroy()
{
    for (auto& v_dispatcher : v2_dispatcher_)
    {
        for (auto& dispatcher : v_dispatcher)
        {
            dispatcher->stop();
        }
    }
    for (auto& v_retriever : v2_retriever_)
    {
        for (auto& retriever : v_retriever)
        {
            retriever.destroy();
        }
    }
}

std::future<RequestResult> BlockManager::submit(Query query)
{
    int partitionId = query.partitionId % cfg_.numPartitions;
    int numToReturn = cfg_.numToReturn;

    // Fire one async call per shard of this partition. Each shard's Dispatcher batches this
    // query together with whatever else concurrently arrives for the same (partition, shard).
    auto v_shardFuture = std::make_shared<std::vector<std::future<RequestResult>>>();
    for (int s = 0; s < cfg_.numShards; s++)
    {
        v_shardFuture->push_back(v2_dispatcher_[partitionId][s]->submit(query));
    }

    // A background task waits for all shard results and merges them. This mirrors the
    // real-world cost of "handling more async threads" as numPartitions x numShards grows: each
    // in-flight request needs its own fan-out/merge task.
    return std::async(std::launch::async,
                      [v_shardFuture, numToReturn]()
                      {
                          std::vector<RequestResult> v_partial;
                          v_partial.reserve(v_shardFuture->size());
                          for (auto& future : *v_shardFuture)
                          {
                              v_partial.push_back(future.get());
                          }
                          return mergeTopK(v_partial, numToReturn);
                      });
}

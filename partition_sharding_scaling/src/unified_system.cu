#include "unified_system.cuh"

void UnifiedSystem::init(SystemConfig cfg, int poolSize)
{
    cfg_ = cfg;
    UnifiedConfig unifiedCfg;
    unifiedCfg.numPartitions   = cfg_.numPartitions;
    unifiedCfg.numShards       = cfg_.numShards;
    unifiedCfg.numDocsPerShard = cfg_.numDocsPerShard;
    unifiedCfg.embDim          = cfg_.embDim;
    embData_.init(unifiedCfg, poolSize);

    int numToReturn = cfg_.numToReturn;
    dispatcher_.start(cfg_.batchSize,
                      cfg_.maxWait,
                      [this, numToReturn](std::vector<Query>& v_query)
                      { return embData_.score(v_query, numToReturn); });
}

void UnifiedSystem::destroy()
{
    dispatcher_.stop();
    embData_.destroy();
}

std::future<RequestResult> UnifiedSystem::submit(Query query) { return dispatcher_.submit(std::move(query)); }

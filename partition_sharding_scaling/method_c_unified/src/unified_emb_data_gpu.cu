#include <unordered_map>

#include "unified_emb_data_gpu.cuh"

void UnifiedEmbDataGpu::init(UnifiedConfig cfg, int poolSize)
{
    cfg_ = cfg;
    vv_retriever_.resize(cfg_.numPartitions);
    for (int p = 0; p < cfg_.numPartitions; p++)
    {
        vv_retriever_[p].resize(cfg_.numShards);
        for (int s = 0; s < cfg_.numShards; s++)
        {
            unsigned seed = (unsigned)(p * cfg_.numShards + s);
            vv_retriever_[p][s].init(cfg_.numDocsPerShard, cfg_.embDim, seed);
        }
    }
    pool_.init(poolSize);
}

void UnifiedEmbDataGpu::destroy()
{
    pool_.destroy();
    for (auto& v_retriever : vv_retriever_)
    {
        for (auto& retriever : v_retriever)
        {
            retriever.destroy();
        }
    }
}

std::vector<RequestResult> UnifiedEmbDataGpu::score(const std::vector<Query>& v_query, int numToReturn)
{
    int                        numQuery = (int)v_query.size();
    std::vector<RequestResult> v_result(numQuery);

    // Group the incoming (possibly multi-partition) batch by partition.
    std::unordered_map<int, std::vector<int>> partitionToIndices; // partitionId -> indices into v_query
    for (int i = 0; i < numQuery; i++)
    {
        partitionToIndices[v_query[i].partitionId % cfg_.numPartitions].push_back(i);
    }

    // Fan out one pooled task per partition present in this batch; the number of OS threads is
    // bounded by the pool size, not by numPartitions x numShards.
    std::vector<std::future<void>> v_future;
    v_future.reserve(partitionToIndices.size());
    for (auto& entry : partitionToIndices)
    {
        int                     partitionId = entry.first;
        const std::vector<int>& v_idx       = entry.second;
        v_future.push_back(pool_.enqueue(
            [this, partitionId, &v_idx, &v_query, &v_result, numToReturn]
            {
                std::vector<Query> v_subQuery;
                v_subQuery.reserve(v_idx.size());
                for (int idx : v_idx)
                {
                    v_subQuery.push_back(v_query[idx]);
                }

                std::vector<std::vector<RequestResult>> v_shardResult(cfg_.numShards);
                for (int s = 0; s < cfg_.numShards; s++)
                {
                    v_shardResult[s] = vv_retriever_[partitionId][s].score(v_subQuery, numToReturn);
                }

                for (size_t i = 0; i < v_idx.size(); i++)
                {
                    std::vector<RequestResult> v_partial;
                    v_partial.reserve(cfg_.numShards);
                    for (int s = 0; s < cfg_.numShards; s++)
                    {
                        v_partial.push_back(v_shardResult[s][i]);
                    }
                    v_result[v_idx[i]] = mergeTopK(v_partial, numToReturn);
                }
            }));
    }
    for (auto& future : v_future)
    {
        future.get();
    }

    return v_result;
}

#pragma once

#include "block_manager.cuh" // reuses SystemConfig
#include "dispatcher.cuh"
#include "unified_emb_data_gpu.cuh"

// Variant C system: exactly one Dispatcher and one (logical) Retriever/EmbDataGpu from the
// caller's perspective. Partition/shard fan-out is fully hidden inside UnifiedEmbDataGpu.
class UnifiedSystem
{
public:
    void init(SystemConfig cfg, int poolSize);
    void destroy();

    std::future<RequestResult> submit(Query query);

private:
    SystemConfig      cfg_;
    UnifiedEmbDataGpu embData_;
    Dispatcher        dispatcher_;
};

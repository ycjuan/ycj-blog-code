#pragma once

#include "dispatcher.cuh"
#include "retrieval_system.cuh"
#include "unified_emb_data_gpu.cuh"

// Variant C system: exactly one Dispatcher and one (logical) Retriever/EmbDataGpu from the
// caller's perspective. Partition/shard fan-out is fully hidden inside UnifiedEmbDataGpu.
class UnifiedSystem : public RetrievalSystem
{
public:
    void init(SystemConfig cfg) override;
    void destroy() override;

    std::future<RequestResult> submit(Query query) override;

private:
    SystemConfig      cfg_;
    UnifiedEmbDataGpu embData_;
    Dispatcher        dispatcher_;
};

#pragma once

#include <cuda_runtime.h>

#include "common.cuh"
#include "emb_data_gpu.cuh"
#include "types.cuh"

// Retriever = "GEMM + topk" compute unit responsible for exactly one (partition, shard) worth
// of documents. It owns its own CUDA stream so multiple Retrievers can be driven concurrently
// (from different CPU threads) without serializing on a single default stream.
class Retriever
{
public:
    void init(int numDocs, int embDim, unsigned seed)
    {
        CHECK_CUDA(cudaStreamCreate(&stream_));
        embData_.init(numDocs, embDim, seed);
    }

    void destroy()
    {
        embData_.destroy();
        if (stream_)
        {
            CHECK_CUDA(cudaStreamDestroy(stream_));
            stream_ = nullptr;
        }
    }

    std::vector<RequestResult> score(const std::vector<Query>& v_query, int numToReturn)
    {
        return embData_.score(v_query, numToReturn, stream_);
    }

private:
    EmbDataGpu   embData_;
    cudaStream_t stream_ = nullptr;
};

#include "backends.hpp"
#include <onnxruntime_cxx_api.h>
#include <cuda_runtime.h>

struct OnnxRuntimeBackend : InferBackend
{
    Ort::Env            env { ORT_LOGGING_LEVEL_WARNING, "jax_compare" };
    Ort::SessionOptions opts;
    Ort::Session        session;
    bool                use_gpu_;

    explicit OnnxRuntimeBackend(const Paths& paths, bool use_gpu)
        : session(nullptr)
        , use_gpu_(use_gpu)
    {
        if (use_gpu)
        {
            OrtCUDAProviderOptions cuda_opts {};
            cuda_opts.device_id = 0;
            opts.AppendExecutionProvider_CUDA(cuda_opts);
        }
        session = Ort::Session(env, paths.onnx_model.c_str(), opts);
    }

    bool supports_device_infer() const override { return use_gpu_; }

    // Kernel-only: inputs/outputs already on device, no H2D/D2H copies.
    void infer_device(const float* d_query,
                      const float* d_docs,
                      float*       d_scores,
                      int          query_dim,
                      int          doc_dim,
                      int          num_docs,
                      int          num_heads) override
    {
        Ort::MemoryInfo cuda_mem("Cuda", OrtDeviceAllocator, 0, OrtMemTypeDefault);

        std::vector<int64_t> query_shape  = { query_dim };
        std::vector<int64_t> docs_shape   = { num_docs, doc_dim };
        std::vector<int64_t> scores_shape = { num_docs, num_heads };

        auto query_val
            = Ort::Value::CreateTensor<float>(cuda_mem, const_cast<float*>(d_query), query_dim, query_shape.data(), 1);
        auto docs_val = Ort::Value::CreateTensor<float>(cuda_mem,
                                                        const_cast<float*>(d_docs),
                                                        (size_t)num_docs * doc_dim,
                                                        docs_shape.data(),
                                                        2);
        auto scores_val
            = Ort::Value::CreateTensor<float>(cuda_mem, d_scores, (size_t)num_docs * num_heads, scores_shape.data(), 2);

        Ort::IoBinding binding(session);
        binding.BindInput("query", query_val);
        binding.BindInput("docs", docs_val);
        binding.BindOutput("scores", scores_val);

        session.Run(Ort::RunOptions { nullptr }, binding);
    }

    std::vector<float> infer(const Input& in) override
    {
        if (use_gpu_)
        {
            float* d_query  = nullptr;
            float* d_docs   = nullptr;
            float* d_scores = nullptr;
            cudaMalloc(&d_query, in.query.size() * sizeof(float));
            cudaMalloc(&d_docs, in.docs.size() * sizeof(float));
            cudaMalloc(&d_scores, (size_t)in.num_docs * in.num_heads * sizeof(float));

            cudaMemcpy(d_query, in.query.data(), in.query.size() * sizeof(float), cudaMemcpyHostToDevice);
            cudaMemcpy(d_docs, in.docs.data(), in.docs.size() * sizeof(float), cudaMemcpyHostToDevice);

            infer_device(d_query, d_docs, d_scores, in.query_dim, in.doc_dim, in.num_docs, in.num_heads);

            std::vector<float> scores(in.num_docs * in.num_heads);
            cudaMemcpy(scores.data(), d_scores, scores.size() * sizeof(float), cudaMemcpyDeviceToHost);

            cudaFree(d_query);
            cudaFree(d_docs);
            cudaFree(d_scores);
            return scores;
        }

        Ort::MemoryInfo mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

        std::vector<int64_t> query_shape = { in.query_dim };
        std::vector<int64_t> docs_shape  = { in.num_docs, in.doc_dim };

        Ort::Value input_tensors[2] = {
            Ort::Value::CreateTensor<float>(mem,
                                            const_cast<float*>(in.query.data()),
                                            in.query.size(),
                                            query_shape.data(),
                                            query_shape.size()),
            Ort::Value::CreateTensor<float>(mem,
                                            const_cast<float*>(in.docs.data()),
                                            in.docs.size(),
                                            docs_shape.data(),
                                            docs_shape.size()),
        };

        const char* input_names[]  = { "query", "docs" };
        const char* output_names[] = { "scores" };

        auto out = session.Run(Ort::RunOptions { nullptr }, input_names, input_tensors, 2, output_names, 1);

        float* data = out[0].GetTensorMutableData<float>();
        return std::vector<float>(data, data + in.num_docs * in.num_heads);
    }
};

std::unique_ptr<InferBackend> make_onnxruntime(const Paths& paths)
{
    return std::make_unique<OnnxRuntimeBackend>(paths, /*use_gpu=*/false);
}

std::unique_ptr<InferBackend> make_onnxruntime_gpu(const Paths& paths)
{
    return std::make_unique<OnnxRuntimeBackend>(paths, /*use_gpu=*/true);
}

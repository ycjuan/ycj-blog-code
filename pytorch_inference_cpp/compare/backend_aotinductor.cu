#include "backends.hpp"
#include <cuda_runtime.h>
#include <torch/csrc/inductor/aoti_package/model_package_loader.h>
#include <torch/torch.h>

struct AOTInductorBackend : InferBackend
{
    torch::inductor::AOTIModelPackageLoader loader;

    explicit AOTInductorBackend(const Paths& paths)
        : loader(paths.aoti_model)
    {
    }

    bool supports_device_infer() const override { return true; }

    void infer_device(const float* d_query,
                      const float* d_docs,
                      float*       d_scores,
                      int          query_dim,
                      int          doc_dim,
                      int          num_docs,
                      int          num_heads) override
    {
        auto opts  = torch::TensorOptions().device(torch::kCUDA).dtype(torch::kFloat32);
        auto query = torch::from_blob(const_cast<float*>(d_query), { query_dim }, opts);
        auto docs  = torch::from_blob(const_cast<float*>(d_docs), { num_docs, doc_dim }, opts);

        auto outputs    = loader.run({ query, docs });
        auto scores_gpu = outputs[0].contiguous(); // [num_docs x num_heads] on GPU
        cudaMemcpy(d_scores,
                   scores_gpu.data_ptr<float>(),
                   num_docs * num_heads * sizeof(float),
                   cudaMemcpyDeviceToDevice);
    }

    std::vector<float> infer(const Input& in) override
    {
        float* d_query  = nullptr;
        float* d_docs   = nullptr;
        float* d_scores = nullptr;
        cudaMalloc(&d_query, in.query.size() * sizeof(float));
        cudaMalloc(&d_docs, in.docs.size() * sizeof(float));
        cudaMalloc(&d_scores, in.num_docs * in.num_heads * sizeof(float));

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
};

std::unique_ptr<InferBackend> make_aotinductor(const Paths& paths)
{
    return std::make_unique<AOTInductorBackend>(paths);
}

#pragma once
#include <memory>
#include <string>
#include <vector>

struct Input
{
    std::vector<float> query; // [query_dim]
    std::vector<float> docs;  // [num_docs x doc_dim], row-major
    int                num_docs;
    int                query_dim;
    int                doc_dim;
    int                num_heads;
};

struct Paths
{
    std::string onnx_model;  // model.onnx
    std::string iree_model;  // model.vmfb
    std::string weights_dir; // weights/
};

// Abstract backend: construct once (loads/compiles the model), call infer() many times.
struct InferBackend
{
    virtual ~InferBackend() = default;

    // End-to-end inference (H2D + kernel + D2H for GPU backends; all-in-one for CPU).
    virtual std::vector<float> infer(const Input& in) = 0;

    // GPU backends implement this: kernel-only, inputs/outputs already on device.
    // d_query: [query_dim], d_docs: [num_docs x doc_dim] (row-major),
    // d_scores: [num_docs x num_heads] output (row-major).
    virtual bool supports_device_infer() const { return false; }
    virtual void infer_device(const float* d_query,
                              const float* d_docs,
                              float*       d_scores,
                              int          query_dim,
                              int          doc_dim,
                              int          num_docs,
                              int          num_heads)
    {
        (void)d_query;
        (void)d_docs;
        (void)d_scores;
        (void)query_dim;
        (void)doc_dim;
        (void)num_docs;
        (void)num_heads;
    }
};

std::unique_ptr<InferBackend> make_onnxruntime(const Paths& paths);
std::unique_ptr<InferBackend> make_onnxruntime_gpu(const Paths& paths);
std::unique_ptr<InferBackend> make_iree(const Paths& paths, const Input& shape_hint);
std::unique_ptr<InferBackend> make_cuda(const Paths& paths, const Input& shape_hint);

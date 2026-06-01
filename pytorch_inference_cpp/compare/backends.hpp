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
    std::string torchscript_model; // model.pt
    std::string onnx_model;        // model.onnx
    std::string weights_dir;       // weights/
};

// Abstract backend: construct once (loads/compiles the model), call infer() many times.
// Separating init from infer lets benchmarks measure pure forward-pass latency.
struct InferBackend
{
    virtual ~InferBackend()                           = default;
    virtual std::vector<float> infer(const Input& in) = 0;
};

std::unique_ptr<InferBackend> make_torchscript(const Paths& paths);
std::unique_ptr<InferBackend> make_onnxruntime(const Paths& paths);
std::unique_ptr<InferBackend> make_tensorrt(const Paths& paths, const Input& profile_input);
std::unique_ptr<InferBackend> make_cuda(const Paths& paths, const Input& shape_hint);

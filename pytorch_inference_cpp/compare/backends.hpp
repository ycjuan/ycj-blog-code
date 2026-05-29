#pragma once
#include <string>
#include <vector>

// Inputs and model locations shared by all backends
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

// Each backend returns scores [num_docs x num_heads], row-major
std::vector<float> run_torchscript(const Paths& paths, const Input& in);
std::vector<float> run_onnxruntime(const Paths& paths, const Input& in);
std::vector<float> run_tensorrt(const Paths& paths, const Input& in);
std::vector<float> run_cuda(const Paths& paths, const Input& in);

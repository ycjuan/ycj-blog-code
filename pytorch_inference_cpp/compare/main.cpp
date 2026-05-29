#include "backends.hpp"
#include <cmath>
#include <iostream>
#include <stdexcept>

static void assertEqual(const std::vector<float>& ref,
                        const std::vector<float>& got,
                        const std::string&        name,
                        float                     tol = 1e-4f)
{
    if (ref.size() != got.size())
        throw std::runtime_error(name + ": size mismatch");
    for (size_t i = 0; i < ref.size(); ++i)
    {
        if (std::fabs(ref[i] - got[i]) > tol)
        {
            throw std::runtime_error(name + ": mismatch at index " + std::to_string(i)
                                     + " (ref=" + std::to_string(ref[i]) + " got=" + std::to_string(got[i]) + ")");
        }
    }
    std::cout << "[PASS] " << name << "\n";
}

static void printScores(const std::string& name, const std::vector<float>& scores, int num_docs, int num_heads)
{
    std::cout << name << ":\n";
    for (int d = 0; d < num_docs; ++d)
    {
        std::cout << "  doc" << d << ": ";
        for (int h = 0; h < num_heads; ++h)
            std::cout << scores[d * num_heads + h] << " ";
        std::cout << "\n";
    }
}

int main()
{
    const int query_dim = 64;
    const int doc_dim   = 128;
    const int num_docs  = 5;
    const int num_heads = 2;

    // Fixed input so all backends are comparable
    std::vector<float> query(query_dim, 0.5f);
    std::vector<float> docs(num_docs * doc_dim);
    for (int i = 0; i < num_docs * doc_dim; ++i)
        docs[i] = static_cast<float>(i % 7) / 7.0f; // non-trivial, not all same

    Input in { query, docs, num_docs, query_dim, doc_dim, num_heads };

    Paths paths {
        "../model.pt",
        "../model.onnx",
        "../weights/",
    };

    std::cout << "Running all backends...\n\n";

    auto scores_ts  = run_torchscript(paths, in);
    auto scores_ort = run_onnxruntime(paths, in);
    auto scores_trt = run_tensorrt(paths, in);
    auto scores_cu  = run_cuda(paths, in);

    std::cout << "\n";
    printScores("TorchScript", scores_ts, num_docs, num_heads);
    printScores("ONNX Runtime", scores_ort, num_docs, num_heads);
    printScores("TensorRT", scores_trt, num_docs, num_heads);
    printScores("Pure CUDA", scores_cu, num_docs, num_heads);

    std::cout << "\nAssertions (tol=1e-4):\n";
    assertEqual(scores_ts, scores_ort, "ONNX Runtime vs TorchScript");
    assertEqual(scores_ts, scores_trt, "TensorRT     vs TorchScript");
    assertEqual(scores_ts, scores_cu, "Pure CUDA    vs TorchScript");

    std::cout << "\nAll backends agree.\n";
    return 0;
}

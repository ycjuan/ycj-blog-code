#include "backends.hpp"
#include <chrono>
#include <cmath>
#include <iostream>
#include <stdexcept>

using Clock = std::chrono::high_resolution_clock;

static double benchMs(InferBackend& backend, const Input& in, int warmup, int iters)
{
    for (int i = 0; i < warmup; ++i)
        backend.infer(in);

    auto t0 = Clock::now();
    for (int i = 0; i < iters; ++i)
        backend.infer(in);
    auto t1 = Clock::now();

    return std::chrono::duration<double, std::milli>(t1 - t0).count() / iters;
}

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
            throw std::runtime_error(name + ": mismatch at index " + std::to_string(i)
                                     + " (ref=" + std::to_string(ref[i]) + " got=" + std::to_string(got[i]) + ")");
    }
    std::cout << "[PASS] " << name << "\n";
}

int main()
{
    const int query_dim = 64;
    const int doc_dim   = 128;
    const int num_docs  = 10000;
    const int num_heads = 2;

    std::vector<float> query(query_dim, 0.5f);
    std::vector<float> docs(num_docs * doc_dim);
    for (int i = 0; i < num_docs * doc_dim; ++i)
        docs[i] = static_cast<float>(i % 7) / 7.0f;

    Input in { query, docs, num_docs, query_dim, doc_dim, num_heads };

    Paths paths { "../model.pt", "../model.onnx", "../weights/" };

    std::cout << "Initializing backends...\n";
    auto ts  = make_torchscript(paths);
    auto ort = make_onnxruntime(paths);
    auto trt = make_tensorrt(paths, in);
    auto cu  = make_cuda(paths, in);

    std::cout << "Checking correctness (num_docs=10000)...\n";
    auto ref     = ts->infer(in);
    auto got_ort = ort->infer(in);
    auto got_trt = trt->infer(in);
    auto got_cu  = cu->infer(in);

    assertEqual(ref, got_ort, "ONNX Runtime vs TorchScript");
    assertEqual(ref, got_trt, "TensorRT     vs TorchScript");
    assertEqual(ref, got_cu, "Pure CUDA    vs TorchScript");

    const int numTrials       = 10;
    const int numWarmupTrials = 3;

    std::cout << "\nBenchmarking (num_docs=10000, " << numWarmupTrials << " warmup + " << numTrials << " trials)...\n";
    printf("  TorchScript  : %7.2f ms\n", benchMs(*ts, in, numWarmupTrials, numTrials));
    printf("  ONNX Runtime : %7.2f ms\n", benchMs(*ort, in, numWarmupTrials, numTrials));
    printf("  TensorRT     : %7.2f ms\n", benchMs(*trt, in, numWarmupTrials, numTrials));
    printf("  Pure CUDA    : %7.2f ms\n", benchMs(*cu, in, numWarmupTrials, numTrials));

    return 0;
}

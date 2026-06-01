#include "backends.hpp"
#include <chrono>
#include <cmath>
#include <cuda_runtime.h>
#include <iostream>
#include <stdexcept>

using Clock = std::chrono::high_resolution_clock;

template <typename Fn>
static double benchMs(Fn fn, int warmup, int iters)
{
    for (int i = 0; i < warmup; ++i)
        fn();
    auto t0 = Clock::now();
    for (int i = 0; i < iters; ++i)
        fn();
    return std::chrono::duration<double, std::milli>(Clock::now() - t0).count() / iters;
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

    Paths paths { "../model.pt", "../model.onnx", "../weights/", "../model.pt2" };

    std::cout << "Initializing backends...\n";
    auto ts   = make_torchscript(paths);
    auto ort  = make_onnxruntime(paths);
    auto trt  = make_tensorrt(paths, in);
    auto cu   = make_cuda(paths, in);
    auto aoti = make_aotinductor(paths);

    std::cout << "Checking correctness (num_docs=" << num_docs << ")...\n";
    auto ref      = ts->infer(in);
    auto got_ort  = ort->infer(in);
    auto got_trt  = trt->infer(in);
    auto got_cu   = cu->infer(in);
    auto got_aoti = aoti->infer(in);

    assertEqual(ref, got_ort, "ONNX Runtime  vs TorchScript");
    assertEqual(ref, got_trt, "TensorRT      vs TorchScript");
    assertEqual(ref, got_cu, "Pure CUDA     vs TorchScript");
    assertEqual(ref, got_aoti, "AOTInductor   vs TorchScript");

    const int numTrials       = 10;
    const int numWarmupTrials = 3;

    // Pre-allocate shared GPU buffers for A/B/C benchmarks
    float* d_query  = nullptr;
    float* d_docs   = nullptr;
    float* d_scores = nullptr;
    cudaMalloc(&d_query, query_dim * sizeof(float));
    cudaMalloc(&d_docs, num_docs * doc_dim * sizeof(float));
    cudaMalloc(&d_scores, num_docs * num_heads * sizeof(float));
    std::vector<float> h_scores(num_docs * num_heads);

    // [A] H2D transfer
    double msA = benchMs(
        [&]()
        {
            cudaMemcpy(d_query, in.query.data(), query_dim * sizeof(float), cudaMemcpyHostToDevice);
            cudaMemcpy(d_docs, in.docs.data(), num_docs * doc_dim * sizeof(float), cudaMemcpyHostToDevice);
        },
        numWarmupTrials,
        numTrials);

    // [C] D2H transfer
    double msC = benchMs(
        [&]() { cudaMemcpy(h_scores.data(), d_scores, num_docs * num_heads * sizeof(float), cudaMemcpyDeviceToHost); },
        numWarmupTrials,
        numTrials);

    // Pre-fill d_query / d_docs once for kernel benchmarks
    cudaMemcpy(d_query, in.query.data(), query_dim * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_docs, in.docs.data(), num_docs * doc_dim * sizeof(float), cudaMemcpyHostToDevice);

    // [B] Kernel-only benchmarks for GPU backends
    double msTrt
        = benchMs([&]() { trt->infer_device(d_query, d_docs, d_scores, query_dim, doc_dim, num_docs, num_heads); },
                  numWarmupTrials,
                  numTrials);
    double msCu
        = benchMs([&]() { cu->infer_device(d_query, d_docs, d_scores, query_dim, doc_dim, num_docs, num_heads); },
                  numWarmupTrials,
                  numTrials);
    double msAoti
        = benchMs([&]() { aoti->infer_device(d_query, d_docs, d_scores, query_dim, doc_dim, num_docs, num_heads); },
                  numWarmupTrials,
                  numTrials);

    // E2E benchmarks for CPU backends (no H2D/D2H separation possible)
    double msTs  = benchMs([&]() { ts->infer(in); }, numWarmupTrials, numTrials);
    double msOrt = benchMs([&]() { ort->infer(in); }, numWarmupTrials, numTrials);

    cudaFree(d_query);
    cudaFree(d_docs);
    cudaFree(d_scores);

    std::cout << "\nBenchmarking (num_docs=" << num_docs << ", " << numWarmupTrials << " warmup + " << numTrials
              << " trials)...\n\n";

    printf("  [A] H2D transfer              : %6.2f ms\n", msA);
    printf("  [C] D2H transfer              : %6.2f ms\n", msC);
    printf("  [A+C] total transfer          : %6.2f ms\n\n", msA + msC);

    printf("  %-14s  e2e: %6.2f ms\n", "TorchScript", msTs);
    printf("  %-14s  e2e: %6.2f ms\n", "ONNX Runtime", msOrt);
    printf("  %-14s  e2e: %6.2f ms  kernel: %6.2f ms\n", "TensorRT", msTrt + msA + msC, msTrt);
    printf("  %-14s  e2e: %6.2f ms  kernel: %6.2f ms\n", "Pure CUDA", msCu + msA + msC, msCu);
    printf("  %-14s  e2e: %6.2f ms  kernel: %6.2f ms\n", "AOTInductor", msAoti + msA + msC, msAoti);

    return 0;
}

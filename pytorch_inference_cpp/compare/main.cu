#include "backends.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cuda_runtime.h>
#include <iostream>
#include <random>
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
    const int num_reqs  = 32; // batch size used for the multi-req benchmark below

    std::mt19937                          rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    std::vector<float> query(num_reqs * query_dim);
    std::generate(query.begin(), query.end(), [&] { return dist(rng); });
    std::vector<float> docs(num_docs * doc_dim);
    std::generate(docs.begin(), docs.end(), [&] { return dist(rng); });

    Input in { query, docs, num_reqs, num_docs, query_dim, doc_dim, num_heads };

    Paths paths { "../model.pt", "../model.onnx", "../weights/", "../model.pt2" };

    std::cout << "Initializing backends...\n";
    auto ts   = make_torchscript(paths);
    auto ort  = make_onnxruntime(paths);
    auto trt  = make_tensorrt(paths, in);
    auto cu   = make_cuda(paths, in);
    auto aoti = make_aotinductor(paths);

    std::cout << "Checking correctness (num_reqs=" << num_reqs << ", num_docs=" << num_docs << ")...\n";
    auto ref      = ts->infer(in);
    auto got_ort  = ort->infer(in);
    auto got_trt  = trt->infer(in);
    auto got_cu   = cu->infer(in);
    auto got_aoti = aoti->infer(in);

    assertEqual(ref, got_ort, "ONNX Runtime  vs TorchScript");
    assertEqual(ref, got_trt, "TensorRT      vs TorchScript");
    assertEqual(ref, got_aoti, "AOTInductor   vs TorchScript");
    assertEqual(ref, got_cu, "Pure CUDA     vs TorchScript");

    const int numTrials       = 10;
    const int numWarmupTrials = 3;

    // Pre-allocate shared GPU buffers for A/B/C benchmarks (sized for the full num_reqs batch)
    float* d_query  = nullptr;
    float* d_docs   = nullptr;
    float* d_scores = nullptr;
    cudaMalloc(&d_query, num_reqs * query_dim * sizeof(float));
    cudaMalloc(&d_docs, num_docs * doc_dim * sizeof(float));
    cudaMalloc(&d_scores, num_reqs * num_docs * num_heads * sizeof(float));
    std::vector<float> h_scores(num_reqs * num_docs * num_heads);

    // [A] H2D transfer
    double msH2D = benchMs(
        [&]()
        {
            cudaMemcpy(d_query, in.query.data(), num_reqs * query_dim * sizeof(float), cudaMemcpyHostToDevice);
            cudaMemcpy(d_docs, in.docs.data(), num_docs * doc_dim * sizeof(float), cudaMemcpyHostToDevice);
        },
        numWarmupTrials,
        numTrials);

    // [C] D2H transfer
    double msD2H = benchMs(
        [&]() {
            cudaMemcpy(h_scores.data(),
                       d_scores,
                       num_reqs * num_docs * num_heads * sizeof(float),
                       cudaMemcpyDeviceToHost);
        },
        numWarmupTrials,
        numTrials);

    // Pre-fill d_query / d_docs once for kernel benchmarks
    cudaMemcpy(d_query, in.query.data(), num_reqs * query_dim * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_docs, in.docs.data(), num_docs * doc_dim * sizeof(float), cudaMemcpyHostToDevice);

    // [B] Kernel-only benchmarks for GPU backends, batched over all num_reqs queries at once
    double msTrt = benchMs(
        [&]() { trt->infer_device(d_query, d_docs, d_scores, num_reqs, query_dim, doc_dim, num_docs, num_heads); },
        numWarmupTrials,
        numTrials);
    double msCu = benchMs(
        [&]() { cu->infer_device(d_query, d_docs, d_scores, num_reqs, query_dim, doc_dim, num_docs, num_heads); },
        numWarmupTrials,
        numTrials);
    double msAoti = benchMs(
        [&]() { aoti->infer_device(d_query, d_docs, d_scores, num_reqs, query_dim, doc_dim, num_docs, num_heads); },
        numWarmupTrials,
        numTrials);

    // [B'] Same GPU backends but scoring one req at a time, looping num_reqs times.
    // Compares against [B] above to quantify the speedup from batching reqs together.
    double msTrtOneByOne = benchMs(
        [&]()
        {
            for (int r = 0; r < num_reqs; ++r)
                trt->infer_device(d_query + r * query_dim,
                                  d_docs,
                                  d_scores,
                                  1,
                                  query_dim,
                                  doc_dim,
                                  num_docs,
                                  num_heads);
        },
        numWarmupTrials,
        numTrials);
    double msCuOneByOne = benchMs(
        [&]()
        {
            for (int r = 0; r < num_reqs; ++r)
                cu->infer_device(d_query + r * query_dim, d_docs, d_scores, 1, query_dim, doc_dim, num_docs, num_heads);
        },
        numWarmupTrials,
        numTrials);
    double msAotiOneByOne = benchMs(
        [&]()
        {
            for (int r = 0; r < num_reqs; ++r)
                aoti->infer_device(d_query + r * query_dim,
                                   d_docs,
                                   d_scores,
                                   1,
                                   query_dim,
                                   doc_dim,
                                   num_docs,
                                   num_heads);
        },
        numWarmupTrials,
        numTrials);

    // E2E benchmarks for CPU backends (no H2D/D2H separation possible)
    double msTs  = benchMs([&]() { ts->infer(in); }, numWarmupTrials, numTrials);
    double msOrt = benchMs([&]() { ort->infer(in); }, numWarmupTrials, numTrials);

    // CPU backends, one req at a time (each call rebuilds a single-req Input view)
    double msTsOneByOne = benchMs(
        [&]()
        {
            for (int r = 0; r < num_reqs; ++r)
            {
                std::vector<float> q(query.begin() + r * query_dim, query.begin() + (r + 1) * query_dim);
                Input              in1 { q, docs, 1, num_docs, query_dim, doc_dim, num_heads };
                ts->infer(in1);
            }
        },
        numWarmupTrials,
        numTrials);
    double msOrtOneByOne = benchMs(
        [&]()
        {
            for (int r = 0; r < num_reqs; ++r)
            {
                std::vector<float> q(query.begin() + r * query_dim, query.begin() + (r + 1) * query_dim);
                Input              in1 { q, docs, 1, num_docs, query_dim, doc_dim, num_heads };
                ort->infer(in1);
            }
        },
        numWarmupTrials,
        numTrials);

    cudaFree(d_query);
    cudaFree(d_docs);
    cudaFree(d_scores);

    std::cout << "\nBenchmarking (num_reqs=" << num_reqs << ", num_docs=" << num_docs << ", " << numWarmupTrials
              << " warmup + " << numTrials << " trials)...\n\n";

    printf("  [A] H2D transfer              : %6.2f ms\n", msH2D);
    printf("  [C] D2H transfer              : %6.2f ms\n", msD2H);
    printf("  [A+C] total transfer          : %6.2f ms\n\n", msH2D + msD2H);

    printf("  %-14s  e2e: %6.2f ms\n", "TorchScript", msTs);
    printf("  %-14s  e2e: %6.2f ms\n", "ONNX Runtime", msOrt);
    printf("  %-14s  e2e: %6.2f ms  kernel: %6.2f ms\n", "TensorRT", msTrt + msH2D + msD2H, msTrt);
    printf("  %-14s  e2e: %6.2f ms  kernel: %6.2f ms\n", "AOTInductor", msAoti + msH2D + msD2H, msAoti);
    printf("  %-14s  e2e: %6.2f ms  kernel: %6.2f ms\n", "Pure CUDA", msCu + msH2D + msD2H, msCu);

    printf("\n  Batched (%d reqs/call) vs one-req-at-a-time (%d calls):\n", num_reqs, num_reqs);
    printf("  %-14s  batched: %7.2f ms   1-by-1: %7.2f ms   speedup: %5.2fx  (kernel-only)\n",
           "TensorRT",
           msTrt,
           msTrtOneByOne,
           msTrtOneByOne / msTrt);
    printf("  %-14s  batched: %7.2f ms   1-by-1: %7.2f ms   speedup: %5.2fx  (kernel-only)\n",
           "AOTInductor",
           msAoti,
           msAotiOneByOne,
           msAotiOneByOne / msAoti);
    printf("  %-14s  batched: %7.2f ms   1-by-1: %7.2f ms   speedup: %5.2fx  (kernel-only)\n",
           "Pure CUDA",
           msCu,
           msCuOneByOne,
           msCuOneByOne / msCu);
    printf("  %-14s  batched: %7.2f ms   1-by-1: %7.2f ms   speedup: %5.2fx  (e2e)\n",
           "TorchScript",
           msTs,
           msTsOneByOne,
           msTsOneByOne / msTs);
    printf("  %-14s  batched: %7.2f ms   1-by-1: %7.2f ms   speedup: %5.2fx  (e2e)\n",
           "ONNX Runtime",
           msOrt,
           msOrtOneByOne,
           msOrtOneByOne / msOrt);

    return 0;
}

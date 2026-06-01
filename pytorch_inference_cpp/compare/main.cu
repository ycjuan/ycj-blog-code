#include "backends.hpp"
#include <chrono>
#include <cmath>
#include <cuda_runtime.h>
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

// Measures the cost of copying query+docs H2D and scores D2H, with no compute.
// This overhead is shared by all GPU backends and is subtracted to get kernel-only time.
static double benchTransferMs(const Input& in, int warmup, int iters)
{
    const size_t query_bytes  = in.query.size() * sizeof(float);
    const size_t docs_bytes   = in.docs.size() * sizeof(float);
    const size_t scores_bytes = in.num_docs * in.num_heads * sizeof(float);

    void* d_query;
    void* d_docs;
    void* d_scores;
    cudaMalloc(&d_query, query_bytes);
    cudaMalloc(&d_docs, docs_bytes);
    cudaMalloc(&d_scores, scores_bytes);

    std::vector<float> h_scores(in.num_docs * in.num_heads);
    auto               doTransfer = [&]()
    {
        cudaMemcpy(d_query, in.query.data(), query_bytes, cudaMemcpyHostToDevice);
        cudaMemcpy(d_docs, in.docs.data(), docs_bytes, cudaMemcpyHostToDevice);
        cudaMemcpy(h_scores.data(), d_scores, scores_bytes, cudaMemcpyDeviceToHost);
    };

    for (int i = 0; i < warmup; ++i)
        doTransfer();

    auto t0 = Clock::now();
    for (int i = 0; i < iters; ++i)
        doTransfer();
    auto t1 = Clock::now();

    cudaFree(d_query);
    cudaFree(d_docs);
    cudaFree(d_scores);

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

    Paths paths { "../model.pt", "../model.onnx", "../weights/", "../model.pt2" };

    std::cout << "Initializing backends...\n";
    auto ts   = make_torchscript(paths);
    auto ort  = make_onnxruntime(paths);
    auto trt  = make_tensorrt(paths, in);
    auto cu   = make_cuda(paths, in);
    auto aoti = make_aotinductor(paths);

    std::cout << "Checking correctness (num_docs=10000)...\n";
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

    const double transferMs = benchTransferMs(in, numWarmupTrials, numTrials);

    std::cout << "\nBenchmarking (num_docs=10000, " << numWarmupTrials << " warmup + " << numTrials << " trials)...\n";
    std::cout << "  H2D + D2H transfer : " << transferMs << " ms (subtracted from GPU backends below)\n\n";

    const double tsMs   = benchMs(*ts, in, numWarmupTrials, numTrials);
    const double ortMs  = benchMs(*ort, in, numWarmupTrials, numTrials);
    const double trtMs  = benchMs(*trt, in, numWarmupTrials, numTrials);
    const double cuMs   = benchMs(*cu, in, numWarmupTrials, numTrials);
    const double aotiMs = benchMs(*aoti, in, numWarmupTrials, numTrials);

    printf("  %-14s  total: %6.2f ms\n", "TorchScript", tsMs);
    printf("  %-14s  total: %6.2f ms\n", "ONNX Runtime", ortMs);
    printf("  %-14s  total: %6.2f ms  kernel: %6.2f ms\n", "TensorRT", trtMs, trtMs - transferMs);
    printf("  %-14s  total: %6.2f ms  kernel: %6.2f ms\n", "Pure CUDA", cuMs, cuMs - transferMs);
    printf("  %-14s  total: %6.2f ms  kernel: %6.2f ms\n", "AOTInductor", aotiMs, aotiMs - transferMs);

    return 0;
}

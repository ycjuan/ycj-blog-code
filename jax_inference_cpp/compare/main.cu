#include "backends.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cuda_runtime.h>
#include <fstream>
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
                        float                     tol = 1e-3f)
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

static bool hasCuda()
{
    int count = 0;
    return cudaGetDeviceCount(&count) == cudaSuccess && count > 0;
}

static bool fileExists(const std::string& path) { return std::ifstream(path).good(); }

int main()
{
    const int query_dim = 64;
    const int doc_dim   = 128;
    const int num_docs  = 10000;
    const int num_heads = 2;

    std::mt19937                          rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    std::vector<float> query(query_dim);
    std::generate(query.begin(), query.end(), [&] { return dist(rng); });
    std::vector<float> docs(num_docs * doc_dim);
    std::generate(docs.begin(), docs.end(), [&] { return dist(rng); });

    Input in { query, docs, num_docs, query_dim, doc_dim, num_heads };
    Paths paths { "../model.onnx", "../model.vmfb", "../model_cuda.vmfb", "../weights/" };

    const bool gpu_available       = hasCuda();
    const bool iree_cuda_available = gpu_available && fileExists(paths.iree_cuda_model);
    if (!gpu_available)
        std::cout << "[INFO] No CUDA GPU detected; skipping GPU backends.\n";
    else if (!iree_cuda_available)
        std::cout << "[INFO] model_cuda.vmfb not found; skipping IREE CUDA backend.\n";

    std::cout << "Initializing backends...\n";
    auto                          ort  = make_onnxruntime(paths);
    auto                          iree = make_iree(paths, in);
    std::unique_ptr<InferBackend> ort_gpu;
    std::unique_ptr<InferBackend> iree_cuda;
    std::unique_ptr<InferBackend> cu;
    if (gpu_available)
    {
        ort_gpu = make_onnxruntime_gpu(paths);
        cu      = make_cuda(paths, in);
        if (iree_cuda_available)
            iree_cuda = make_iree_cuda(paths, in);
    }

    std::cout << "Checking correctness (num_docs=" << num_docs << ")...\n";
    auto ref = ort->infer(in);
    assertEqual(ref, iree->infer(in), "IREE (CPU)      vs ORT (CPU)");
    if (ort_gpu)
        assertEqual(ref, ort_gpu->infer(in), "ORT (GPU)       vs ORT (CPU)");
    if (iree_cuda)
        assertEqual(ref, iree_cuda->infer(in), "IREE (CUDA)     vs ORT (CPU)");
    if (cu)
        assertEqual(ref, cu->infer(in), "Pure CUDA       vs ORT (CPU)");

    const int numTrials       = 10;
    const int numWarmupTrials = 3;

    double msOrt  = benchMs([&]() { ort->infer(in); }, numWarmupTrials, numTrials);
    double msIree = benchMs([&]() { iree->infer(in); }, numWarmupTrials, numTrials);

    std::cout << "\nBenchmarking (num_docs=" << num_docs << ", " << numWarmupTrials << " warmup + " << numTrials
              << " trials)...\n\n";

    printf("  %-25s  e2e: %6.2f ms\n", "ONNX Runtime (CPU)", msOrt);
    printf("  %-25s  e2e: %6.2f ms\n", "IREE (CPU, local-sync)", msIree);

    if (gpu_available)
    {
        float* d_query  = nullptr;
        float* d_docs   = nullptr;
        float* d_scores = nullptr;
        cudaMalloc(&d_query, query_dim * sizeof(float));
        cudaMalloc(&d_docs, num_docs * doc_dim * sizeof(float));
        cudaMalloc(&d_scores, num_docs * num_heads * sizeof(float));
        std::vector<float> h_scores(num_docs * num_heads);

        double msH2D = benchMs(
            [&]()
            {
                cudaMemcpy(d_query, in.query.data(), query_dim * sizeof(float), cudaMemcpyHostToDevice);
                cudaMemcpy(d_docs, in.docs.data(), num_docs * doc_dim * sizeof(float), cudaMemcpyHostToDevice);
            },
            numWarmupTrials,
            numTrials);

        double msD2H = benchMs(
            [&]()
            { cudaMemcpy(h_scores.data(), d_scores, num_docs * num_heads * sizeof(float), cudaMemcpyDeviceToHost); },
            numWarmupTrials,
            numTrials);

        // Pre-copy inputs so kernel benchmarks start with data already on device.
        cudaMemcpy(d_query, in.query.data(), query_dim * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_docs, in.docs.data(), num_docs * doc_dim * sizeof(float), cudaMemcpyHostToDevice);

        double msOrtGpu = benchMs(
            [&]() { ort_gpu->infer_device(d_query, d_docs, d_scores, query_dim, doc_dim, num_docs, num_heads); },
            numWarmupTrials,
            numTrials);

        double msIreeCuda = 0;
        if (iree_cuda)
            msIreeCuda = benchMs(
                [&]() { iree_cuda->infer_device(d_query, d_docs, d_scores, query_dim, doc_dim, num_docs, num_heads); },
                numWarmupTrials,
                numTrials);

        double msCu
            = benchMs([&]() { cu->infer_device(d_query, d_docs, d_scores, query_dim, doc_dim, num_docs, num_heads); },
                      numWarmupTrials,
                      numTrials);

        printf("  [A] H2D transfer              :   %5.2f ms\n", msH2D);
        printf("  [C] D2H transfer              :   %5.2f ms\n", msD2H);
        printf("  [A+C] total transfer          :   %5.2f ms\n\n", msH2D + msD2H);

        printf("  %-25s  e2e: %6.2f ms  kernel: %6.2f ms\n", "ONNX Runtime (GPU)", msOrtGpu + msH2D + msD2H, msOrtGpu);
        if (iree_cuda)
            printf("  %-25s  e2e: %6.2f ms  kernel: %6.2f ms\n", "IREE (CUDA)", msIreeCuda + msH2D + msD2H, msIreeCuda);
        printf("  %-25s  e2e: %6.2f ms  kernel: %6.2f ms\n", "Pure CUDA", msCu + msH2D + msD2H, msCu);

        cudaFree(d_query);
        cudaFree(d_docs);
        cudaFree(d_scores);
    }

    return 0;
}

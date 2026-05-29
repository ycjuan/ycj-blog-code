#include "backends.hpp"
#include <NvInfer.h>
#include <NvOnnxParser.h>
#include <cuda_runtime_api.h>
#include <iostream>
#include <memory>
#include <stdexcept>

namespace
{
class Logger : public nvinfer1::ILogger
{
public:
    void log(Severity severity, const char* msg) noexcept override
    {
        if (severity <= Severity::kWARNING)
            std::cerr << "[TRT] " << msg << "\n";
    }
} gLogger;
} // namespace

std::vector<float> run_tensorrt(const Paths& paths, const Input& in)
{
    std::unique_ptr<nvinfer1::IBuilder>           builder(nvinfer1::createInferBuilder(gLogger));
    std::unique_ptr<nvinfer1::INetworkDefinition> network(builder->createNetworkV2(0));
    std::unique_ptr<nvonnxparser::IParser>        parser(nvonnxparser::createParser(*network, gLogger));

    parser->parseFromFile(paths.onnx_model.c_str(), static_cast<int>(nvinfer1::ILogger::Severity::kWARNING));

    std::unique_ptr<nvinfer1::IBuilderConfig> config(builder->createBuilderConfig());
    config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, 1 << 20);

    nvinfer1::IOptimizationProfile* profile = builder->createOptimizationProfile();
    profile->setDimensions("query", nvinfer1::OptProfileSelector::kMIN, nvinfer1::Dims { 1, { in.query_dim } });
    profile->setDimensions("query", nvinfer1::OptProfileSelector::kOPT, nvinfer1::Dims { 1, { in.query_dim } });
    profile->setDimensions("query", nvinfer1::OptProfileSelector::kMAX, nvinfer1::Dims { 1, { in.query_dim } });
    profile->setDimensions("docs", nvinfer1::OptProfileSelector::kMIN, nvinfer1::Dims2(1, in.doc_dim));
    profile->setDimensions("docs", nvinfer1::OptProfileSelector::kOPT, nvinfer1::Dims2(in.num_docs, in.doc_dim));
    profile->setDimensions("docs", nvinfer1::OptProfileSelector::kMAX, nvinfer1::Dims2(100, in.doc_dim));
    config->addOptimizationProfile(profile);

    std::unique_ptr<nvinfer1::ICudaEngine>       engine(builder->buildEngineWithConfig(*network, *config));
    std::unique_ptr<nvinfer1::IExecutionContext> context(engine->createExecutionContext());

    context->setInputShape("query", nvinfer1::Dims { 1, { in.query_dim } });
    context->setInputShape("docs", nvinfer1::Dims2(in.num_docs, in.doc_dim));

    void* d_query  = nullptr;
    void* d_docs   = nullptr;
    void* d_scores = nullptr;
    cudaMalloc(&d_query, in.query.size() * sizeof(float));
    cudaMalloc(&d_docs, in.docs.size() * sizeof(float));
    cudaMalloc(&d_scores, in.num_docs * in.num_heads * sizeof(float));

    cudaMemcpy(d_query, in.query.data(), in.query.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_docs, in.docs.data(), in.docs.size() * sizeof(float), cudaMemcpyHostToDevice);

    context->setTensorAddress("query", d_query);
    context->setTensorAddress("docs", d_docs);
    context->setTensorAddress("scores", d_scores);
    context->enqueueV3(0);
    cudaDeviceSynchronize();

    std::vector<float> scores(in.num_docs * in.num_heads);
    cudaMemcpy(scores.data(), d_scores, scores.size() * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_query);
    cudaFree(d_docs);
    cudaFree(d_scores);
    return scores;
}

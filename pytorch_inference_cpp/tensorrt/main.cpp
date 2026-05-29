#include <NvInfer.h>
#include <NvOnnxParser.h>
#include <cuda_runtime_api.h>
#include <iostream>
#include <memory>
#include <vector>

class Logger : public nvinfer1::ILogger
{
public:
    void log(Severity severity, const char* msg) noexcept override
    {
        if (severity <= Severity::kWARNING)
            std::cerr << "[TRT] " << msg << "\n";
    }
} gLogger;

template <typename T>
using TrtUniquePtr = std::unique_ptr<T>;

int main()
{
    // --- Build engine from ONNX ---
    TrtUniquePtr<nvinfer1::IBuilder>           builder(nvinfer1::createInferBuilder(gLogger));
    TrtUniquePtr<nvinfer1::INetworkDefinition> network(builder->createNetworkV2(0));
    TrtUniquePtr<nvonnxparser::IParser>        parser(nvonnxparser::createParser(*network, gLogger));

    parser->parseFromFile("../model.onnx", static_cast<int>(nvinfer1::ILogger::Severity::kWARNING));

    TrtUniquePtr<nvinfer1::IBuilderConfig> config(builder->createBuilderConfig());
    config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, 1 << 20);

    // query is static [64]; docs has dynamic num_docs dim
    nvinfer1::IOptimizationProfile* profile = builder->createOptimizationProfile();
    profile->setDimensions("query", nvinfer1::OptProfileSelector::kMIN, nvinfer1::Dims { 1, { 64 } });
    profile->setDimensions("query", nvinfer1::OptProfileSelector::kOPT, nvinfer1::Dims { 1, { 64 } });
    profile->setDimensions("query", nvinfer1::OptProfileSelector::kMAX, nvinfer1::Dims { 1, { 64 } });
    profile->setDimensions("docs", nvinfer1::OptProfileSelector::kMIN, nvinfer1::Dims2(1, 128));
    profile->setDimensions("docs", nvinfer1::OptProfileSelector::kOPT, nvinfer1::Dims2(5, 128));
    profile->setDimensions("docs", nvinfer1::OptProfileSelector::kMAX, nvinfer1::Dims2(100, 128));
    config->addOptimizationProfile(profile);

    TrtUniquePtr<nvinfer1::ICudaEngine>       engine(builder->buildEngineWithConfig(*network, *config));
    TrtUniquePtr<nvinfer1::IExecutionContext> context(engine->createExecutionContext());

    // --- Set dynamic shapes for this call: 5 docs ---
    const int num_docs = 5;
    context->setInputShape("query", nvinfer1::Dims { 1, { 64 } });
    context->setInputShape("docs", nvinfer1::Dims2(num_docs, 128));

    // --- Allocate GPU buffers ---
    std::vector<float> h_query(64, 0.5f);
    std::vector<float> h_docs(num_docs * 128, 0.5f);
    std::vector<float> h_scores(num_docs * 2);

    void* d_query  = nullptr;
    void* d_docs   = nullptr;
    void* d_scores = nullptr;
    cudaMalloc(&d_query, h_query.size() * sizeof(float));
    cudaMalloc(&d_docs, h_docs.size() * sizeof(float));
    cudaMalloc(&d_scores, h_scores.size() * sizeof(float));

    cudaMemcpy(d_query, h_query.data(), h_query.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_docs, h_docs.data(), h_docs.size() * sizeof(float), cudaMemcpyHostToDevice);

    context->setTensorAddress("query", d_query);
    context->setTensorAddress("docs", d_docs);
    context->setTensorAddress("scores", d_scores);

    // TRT 10+ named-tensor API: setTensorAddress + enqueueV3 (replaces executeV2)
    context->enqueueV3(0);

    cudaMemcpy(h_scores.data(), d_scores, h_scores.size() * sizeof(float), cudaMemcpyDeviceToHost);

    std::cout << "Scores shape: [" << num_docs << ", 2]\n";
    std::cout << "Scores:\n";
    for (int d = 0; d < num_docs; ++d)
        std::cout << h_scores[d * 2] << " " << h_scores[d * 2 + 1] << "\n";

    cudaFree(d_query);
    cudaFree(d_docs);
    cudaFree(d_scores);
    return 0;
}

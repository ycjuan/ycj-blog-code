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

struct TensorRTBackend : InferBackend
{
    std::unique_ptr<nvinfer1::ICudaEngine>       engine;
    std::unique_ptr<nvinfer1::IExecutionContext> context;
    int                                          query_dim, doc_dim, num_heads;

    TensorRTBackend(const Paths& paths, const Input& profile_input)
        : query_dim(profile_input.query_dim)
        , doc_dim(profile_input.doc_dim)
        , num_heads(profile_input.num_heads)
    {
        std::unique_ptr<nvinfer1::IBuilder>           builder(nvinfer1::createInferBuilder(gLogger));
        std::unique_ptr<nvinfer1::INetworkDefinition> network(builder->createNetworkV2(0));
        std::unique_ptr<nvonnxparser::IParser>        parser(nvonnxparser::createParser(*network, gLogger));

        parser->parseFromFile(paths.onnx_model.c_str(), static_cast<int>(nvinfer1::ILogger::Severity::kWARNING));

        std::unique_ptr<nvinfer1::IBuilderConfig> config(builder->createBuilderConfig());
        config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, 256 << 20);

        const int                       max_docs = profile_input.num_docs;
        nvinfer1::IOptimizationProfile* profile  = builder->createOptimizationProfile();
        profile->setDimensions("query", nvinfer1::OptProfileSelector::kMIN, nvinfer1::Dims { 1, { query_dim } });
        profile->setDimensions("query", nvinfer1::OptProfileSelector::kOPT, nvinfer1::Dims { 1, { query_dim } });
        profile->setDimensions("query", nvinfer1::OptProfileSelector::kMAX, nvinfer1::Dims { 1, { query_dim } });
        profile->setDimensions("docs", nvinfer1::OptProfileSelector::kMIN, nvinfer1::Dims2(1, doc_dim));
        profile->setDimensions("docs", nvinfer1::OptProfileSelector::kOPT, nvinfer1::Dims2(max_docs, doc_dim));
        profile->setDimensions("docs", nvinfer1::OptProfileSelector::kMAX, nvinfer1::Dims2(max_docs, doc_dim));
        config->addOptimizationProfile(profile);

        engine.reset(builder->buildEngineWithConfig(*network, *config));
        context.reset(engine->createExecutionContext());
    }

    bool supports_device_infer() const override { return true; }

    void infer_device(const float* d_query,
                      const float* d_docs,
                      float*       d_scores,
                      int          query_dim_,
                      int          doc_dim_,
                      int          num_docs,
                      int          num_heads_) override
    {
        context->setInputShape("query", nvinfer1::Dims { 1, { query_dim_ } });
        context->setInputShape("docs", nvinfer1::Dims2(num_docs, doc_dim_));
        context->setTensorAddress("query", const_cast<float*>(d_query));
        context->setTensorAddress("docs", const_cast<float*>(d_docs));
        context->setTensorAddress("scores", d_scores);
        context->enqueueV3(0);
        cudaDeviceSynchronize();
    }

    std::vector<float> infer(const Input& in) override
    {
        void* d_query  = nullptr;
        void* d_docs   = nullptr;
        void* d_scores = nullptr;
        cudaMalloc(&d_query, in.query.size() * sizeof(float));
        cudaMalloc(&d_docs, in.docs.size() * sizeof(float));
        cudaMalloc(&d_scores, in.num_docs * in.num_heads * sizeof(float));

        cudaMemcpy(d_query, in.query.data(), in.query.size() * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_docs, in.docs.data(), in.docs.size() * sizeof(float), cudaMemcpyHostToDevice);

        infer_device(static_cast<float*>(d_query),
                     static_cast<float*>(d_docs),
                     static_cast<float*>(d_scores),
                     in.query_dim,
                     in.doc_dim,
                     in.num_docs,
                     in.num_heads);

        std::vector<float> scores(in.num_docs * in.num_heads);
        cudaMemcpy(scores.data(), d_scores, scores.size() * sizeof(float), cudaMemcpyDeviceToHost);

        cudaFree(d_query);
        cudaFree(d_docs);
        cudaFree(d_scores);
        return scores;
    }
};

std::unique_ptr<InferBackend> make_tensorrt(const Paths& paths, const Input& profile_input)
{
    return std::make_unique<TensorRTBackend>(paths, profile_input);
}

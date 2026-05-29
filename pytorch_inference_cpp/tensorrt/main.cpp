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
    // --- Build engine from ONNX (run once; in production, serialize and cache it) ---
    TrtUniquePtr<nvinfer1::IBuilder>           builder(nvinfer1::createInferBuilder(gLogger));
    TrtUniquePtr<nvinfer1::INetworkDefinition> network(builder->createNetworkV2(0));
    TrtUniquePtr<nvonnxparser::IParser>        parser(nvonnxparser::createParser(*network, gLogger));

    parser->parseFromFile("../model.onnx", static_cast<int>(nvinfer1::ILogger::Severity::kWARNING));

    TrtUniquePtr<nvinfer1::IBuilderConfig> config(builder->createBuilderConfig());
    config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, 1 << 20);

    TrtUniquePtr<nvinfer1::ICudaEngine> engine(builder->buildEngineWithConfig(*network, *config));

    // --- Run inference ---
    TrtUniquePtr<nvinfer1::IExecutionContext> context(engine->createExecutionContext());

    float h_input[4]  = { 1.0f, 2.0f, 3.0f, 4.0f };
    float h_output[2] = {};

    void* d_input  = nullptr;
    void* d_output = nullptr;
    cudaMalloc(&d_input, 4 * sizeof(float));
    cudaMalloc(&d_output, 2 * sizeof(float));

    cudaMemcpy(d_input, h_input, 4 * sizeof(float), cudaMemcpyHostToDevice);

    void* bindings[] = { d_input, d_output };
    context->executeV2(bindings);

    cudaMemcpy(h_output, d_output, 2 * sizeof(float), cudaMemcpyDeviceToHost);

    std::cout << "Output: [" << h_output[0] << ", " << h_output[1] << "]\n";

    cudaFree(d_input);
    cudaFree(d_output);
    return 0;
}

#include <NvInfer.h>
#include <NvOnnxParser.h>
#include <cuda_runtime_api.h>
#include <fstream>
#include <iostream>
#include <memory>
#include <vector>

// TensorRT logger
class Logger : public nvinfer1::ILogger
{
public:
    void log(Severity severity, const char* msg) noexcept override
    {
        if (severity <= Severity::kWARNING)
            std::cerr << "[TRT] " << msg << "\n";
    }
} gLogger;

// Deleter for TensorRT objects (they use destroy() instead of delete)
struct TrtDeleter
{
    template <typename T>
    void operator()(T* obj) const
    {
        obj->destroy();
    }
};
template <typename T>
using TrtUniquePtr = std::unique_ptr<T, TrtDeleter>;

std::vector<char> loadEngineFile(const std::string& path)
{
    std::ifstream   file(path, std::ios::binary | std::ios::ate);
    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);
    std::vector<char> buffer(size);
    file.read(buffer.data(), size);
    return buffer;
}

int main()
{
    // --- Build engine from ONNX (run once; in production, serialize and cache it) ---
    TrtUniquePtr<nvinfer1::IBuilder> builder(nvinfer1::createInferBuilder(gLogger));
    const auto explicitBatch = 1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    TrtUniquePtr<nvinfer1::INetworkDefinition> network(builder->createNetworkV2(explicitBatch));
    TrtUniquePtr<nvonnxparser::IParser>        parser(nvonnxparser::createParser(*network, gLogger));

    parser->parseFromFile("../model.onnx", static_cast<int>(nvinfer1::ILogger::Severity::kWARNING));

    TrtUniquePtr<nvinfer1::IBuilderConfig> config(builder->createBuilderConfig());
    config->setMaxWorkspaceSize(1 << 20); // 1 MB

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

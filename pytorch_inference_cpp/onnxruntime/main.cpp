#include <iostream>
#include <onnxruntime_cxx_api.h>
#include <vector>

int main()
{
    Ort::Env            env(ORT_LOGGING_LEVEL_WARNING, "onnxruntime_inference");
    Ort::SessionOptions session_options;

    Ort::Session session(env, "../model.onnx", session_options);

    Ort::AllocatorWithDefaultOptions allocator;

    const char* input_name  = "input";
    const char* output_name = "output";

    std::vector<float>   input_data  = { 1.0f, 2.0f, 3.0f, 4.0f };
    std::vector<int64_t> input_shape = { 1, 4 };

    Ort::MemoryInfo memory_info  = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value      input_tensor = Ort::Value::CreateTensor<float>(memory_info,
                                                              input_data.data(),
                                                              input_data.size(),
                                                              input_shape.data(),
                                                              input_shape.size());

    std::vector<const char*> input_names  = { input_name };
    std::vector<const char*> output_names = { output_name };

    auto output_tensors
        = session.Run(Ort::RunOptions { nullptr }, input_names.data(), &input_tensor, 1, output_names.data(), 1);

    float* output_data  = output_tensors[0].GetTensorMutableData<float>();
    auto   output_shape = output_tensors[0].GetTensorTypeAndShapeInfo().GetShape();

    std::cout << "Output shape: [" << output_shape[0] << ", " << output_shape[1] << "]\n";
    std::cout << "Output: [" << output_data[0] << ", " << output_data[1] << "]\n";

    return 0;
}

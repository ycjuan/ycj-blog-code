#include <iostream>
#include <onnxruntime_cxx_api.h>
#include <vector>

int main()
{
    Ort::Env            env(ORT_LOGGING_LEVEL_WARNING, "onnxruntime_inference");
    Ort::SessionOptions session_options;

    Ort::Session session(env, "../model.onnx", session_options);

    // query: [64], docs: [5, 128]
    std::vector<float>   query_data(64);
    std::vector<float>   docs_data(5 * 128);
    std::vector<int64_t> query_shape = { 64 };
    std::vector<int64_t> docs_shape  = { 5, 128 };

    // Fill with dummy data
    for (auto& v : query_data)
        v = 0.5f;
    for (auto& v : docs_data)
        v = 0.5f;

    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

    Ort::Value input_tensors[2] = {
        Ort::Value::CreateTensor<float>(memory_info,
                                        query_data.data(),
                                        query_data.size(),
                                        query_shape.data(),
                                        query_shape.size()),
        Ort::Value::CreateTensor<float>(memory_info,
                                        docs_data.data(),
                                        docs_data.size(),
                                        docs_shape.data(),
                                        docs_shape.size()),
    };

    const char* input_names[]  = { "query", "docs" };
    const char* output_names[] = { "scores" };

    auto output_tensors = session.Run(Ort::RunOptions { nullptr }, input_names, input_tensors, 2, output_names, 1);

    float* scores    = output_tensors[0].GetTensorMutableData<float>();
    auto   out_shape = output_tensors[0].GetTensorTypeAndShapeInfo().GetShape();

    std::cout << "Scores shape: [" << out_shape[0] << ", " << out_shape[1] << "]\n";
    std::cout << "Scores:\n";
    for (int d = 0; d < out_shape[0]; ++d)
    {
        for (int h = 0; h < out_shape[1]; ++h)
            std::cout << scores[d * out_shape[1] + h] << " ";
        std::cout << "\n";
    }

    return 0;
}

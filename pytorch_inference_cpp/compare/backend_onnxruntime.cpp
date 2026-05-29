#include "backends.hpp"
#include <onnxruntime_cxx_api.h>

struct OnnxRuntimeBackend : InferBackend
{
    Ort::Env            env { ORT_LOGGING_LEVEL_WARNING, "compare" };
    Ort::SessionOptions opts;
    Ort::Session        session;

    explicit OnnxRuntimeBackend(const Paths& paths)
        : session(env, paths.onnx_model.c_str(), opts)
    {
    }

    std::vector<float> infer(const Input& in) override
    {
        Ort::MemoryInfo mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

        std::vector<int64_t> query_shape = { in.query_dim };
        std::vector<int64_t> docs_shape  = { in.num_docs, in.doc_dim };

        Ort::Value input_tensors[2] = {
            Ort::Value::CreateTensor<float>(mem,
                                            const_cast<float*>(in.query.data()),
                                            in.query.size(),
                                            query_shape.data(),
                                            query_shape.size()),
            Ort::Value::CreateTensor<float>(mem,
                                            const_cast<float*>(in.docs.data()),
                                            in.docs.size(),
                                            docs_shape.data(),
                                            docs_shape.size()),
        };

        const char* input_names[]  = { "query", "docs" };
        const char* output_names[] = { "scores" };

        auto out = session.Run(Ort::RunOptions { nullptr }, input_names, input_tensors, 2, output_names, 1);

        float* data = out[0].GetTensorMutableData<float>();
        return std::vector<float>(data, data + in.num_docs * in.num_heads);
    }
};

std::unique_ptr<InferBackend> make_onnxruntime(const Paths& paths) { return std::make_unique<OnnxRuntimeBackend>(paths); }

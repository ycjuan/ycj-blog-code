#include "backends.hpp"
#include <torch/script.h>

struct TorchScriptBackend : InferBackend
{
    torch::jit::script::Module model;

    explicit TorchScriptBackend(const Paths& paths)
    {
        model = torch::jit::load(paths.torchscript_model);
        model.eval();
    }

    std::vector<float> infer(const Input& in) override
    {
        auto query = torch::from_blob(const_cast<float*>(in.query.data()), { in.query_dim }, torch::kFloat32).clone();
        auto docs  = torch::from_blob(const_cast<float*>(in.docs.data()), { in.num_docs, in.doc_dim }, torch::kFloat32)
                        .clone();

        std::vector<torch::jit::IValue> inputs = { query, docs };
        auto                            scores = model.forward(inputs).toTensor().contiguous();
        return std::vector<float>(scores.data_ptr<float>(), scores.data_ptr<float>() + scores.numel());
    }
};

std::unique_ptr<InferBackend> make_torchscript(const Paths& paths)
{
    return std::make_unique<TorchScriptBackend>(paths);
}

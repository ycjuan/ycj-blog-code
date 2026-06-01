#include "backends.hpp"
#include <torch/csrc/inductor/aoti_package/model_package_loader.h>
#include <torch/torch.h>

struct AOTInductorBackend : InferBackend
{
    torch::inductor::AOTIModelPackageLoader loader;

    explicit AOTInductorBackend(const Paths& paths)
        : loader(paths.aoti_model)
    {
    }

    std::vector<float> infer(const Input& in) override
    {
        auto opts = torch::TensorOptions().device(torch::kCUDA).dtype(torch::kFloat32);
        auto query
            = torch::from_blob(const_cast<float*>(in.query.data()), { in.query_dim }, torch::kFloat32).clone().to(opts);
        auto docs = torch::from_blob(const_cast<float*>(in.docs.data()), { in.num_docs, in.doc_dim }, torch::kFloat32)
                        .clone()
                        .to(opts);

        auto outputs = loader.run({ query, docs });
        auto scores  = outputs[0].cpu().contiguous();
        return std::vector<float>(scores.data_ptr<float>(), scores.data_ptr<float>() + scores.numel());
    }
};

std::unique_ptr<InferBackend> make_aotinductor(const Paths& paths)
{
    return std::make_unique<AOTInductorBackend>(paths);
}

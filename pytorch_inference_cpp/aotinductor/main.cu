#include <iostream>
#include <torch/csrc/inductor/aoti_package/model_package_loader.h>
#include <torch/torch.h>

int main()
{
    torch::inductor::AOTIModelPackageLoader loader("../model.pt2");

    const int query_dim = 64;
    const int doc_dim   = 128;
    const int num_docs  = 5;

    auto opts  = torch::TensorOptions().device(torch::kCUDA).dtype(torch::kFloat32);
    auto query = torch::randn({ query_dim }, opts);
    auto docs  = torch::randn({ num_docs, doc_dim }, opts);

    auto outputs = loader.run({ query, docs });
    auto scores  = outputs[0].cpu();

    std::cout << "Scores shape: [" << scores.size(0) << ", " << scores.size(1) << "]\n";
    std::cout << "Scores:\n" << scores << "\n";

    return 0;
}

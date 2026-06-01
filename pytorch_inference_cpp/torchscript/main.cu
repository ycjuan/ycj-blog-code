#include <iostream>
#include <torch/script.h>

int main()
{
    torch::jit::script::Module model;
    try
    {
        model = torch::jit::load("../model.pt");
    }
    catch (const c10::Error& e)
    {
        std::cerr << "Failed to load model: " << e.what() << "\n";
        return 1;
    }
    model.eval();

    torch::Tensor query = torch::randn({ 64 });
    torch::Tensor docs  = torch::randn({ 5, 128 });

    std::vector<torch::jit::IValue> inputs = { query, docs };
    torch::Tensor                   scores = model.forward(inputs).toTensor();

    std::cout << "Scores shape: [" << scores.size(0) << ", " << scores.size(1) << "]\n";
    std::cout << "Scores:\n" << scores << "\n";

    return 0;
}

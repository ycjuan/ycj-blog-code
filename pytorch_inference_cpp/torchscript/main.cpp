#include <iostream>
#include <torch/script.h>
#include <vector>

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

    torch::Tensor input = torch::randn({ 1, 4 });
    std::cout << "Input:  " << input << "\n";

    std::vector<torch::jit::IValue> inputs = { input };
    torch::Tensor                   output = model.forward(inputs).toTensor();
    std::cout << "Output: " << output << "\n";

    return 0;
}

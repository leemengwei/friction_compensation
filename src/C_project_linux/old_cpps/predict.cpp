
#include <torch/script.h>

//Function to run
at::Tensor predict(torch::jit::script::Module model, std::vector<torch::jit::IValue> inputs) {
  at::Tensor output;
  output = model.forward(inputs).toTensor();
  return output;
}



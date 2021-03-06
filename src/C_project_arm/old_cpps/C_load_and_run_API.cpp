
#include <iostream>
#include <memory>

#include <torch/script.h>


at::Tensor predict(torch::jit::script::Module model, std::vector<torch::jit::IValue> inputs);
std::vector<torch::jit::IValue> get_data(bool use_cuda);
torch::jit::script::Module get_model(const char* model_path, bool use_cuda);

int main(int argc, const char* argv[]) {
  //Config:
  bool use_cuda=(argc>1)?true:false;
  if (use_cuda){std::cout<<"USING GPU"<<std::endl;}
  else{std::cout<<"USING CPU"<<std::endl;}

  //Get model:
  const char* model_path="../models/NN_weights_uniform_C.pt";
  torch::jit::script::Module model=get_model(model_path, use_cuda);

  //Get data:
  std::vector<torch::jit::IValue> inputs=get_data(use_cuda);

  //Forward run loop:
  at::Tensor output;
  output = predict(model, inputs);

  //Report:
  std::cout << inputs << std::endl;
  std::cout << output.slice(/*dim=*/1, /*start=*/0, /*end=*/99) << '\n';
}




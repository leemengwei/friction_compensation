
#include <torch/script.h>

//Function to load model
torch::jit::script::Module get_model(const char* model_path, bool use_cuda) {
  torch::jit::script::Module module;
  try {
    module = torch::jit::load(model_path);
  }
  catch (const c10::Error& e) {
    std::cerr << "Error loading the model\n";
    exit(-1);
  }
  if (use_cuda){
    module.to(at::kCUDA);
  }
  return module;
}



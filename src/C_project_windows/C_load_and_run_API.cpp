#define num_of_cols 25
#define num_of_rows 1

#include <iostream>
#include <memory>

#include <torch/script.h>

//at::Tensor predict(torch::jit::script::Module model, std::vector<torch::jit::IValue> inputs);
//std::vector<torch::jit::IValue> get_data(bool use_cuda);
//torch::jit::script::Module get_model(const char* model_path, bool use_cuda);


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

//Function to convert data
std::vector<torch::jit::IValue> convert_data(bool use_cuda, float flat[][num_of_cols]) {

  at::Tensor data = torch::from_blob(flat, {num_of_rows,num_of_cols}, torch::kFloat);
  data = data.toType(torch::kFloat);
  if (use_cuda){
    data=data.to(at::kCUDA);
   }
  std::vector<torch::jit::IValue> inputs;
  inputs.push_back(data);
  return inputs;
}

//Function to run
at::Tensor predict(bool use_cuda, torch::jit::script::Module model, float flat[][num_of_cols]) {
  //Convert
  std::vector<torch::jit::IValue> inputs = convert_data(use_cuda, flat);
  //and run
  at::Tensor output;
  output = model.forward(inputs).toTensor();
  return output;
}

//Function to preprocess input data
void preprocess_data(float flat[][num_of_cols]) {
    float mean[num_of_rows][num_of_cols] = {
        2.51065213e-04, 2.26194300e+02, -3.87426864e-03, -3.87426074e-03, 2.22841566e-04, 3.23421159e+01, -3.59547080e-01, -3.30278722e-03, -6.44383032e-04, 8.86972255e+01, 8.88570940e-02, 2.07215009e-03, -5.10639766e-02, -1.07646179e+00, -5.16060366e-03, -1.96551706e-05, -3.95824362e-04, -4.44822192e-01, 1.01560208e-03, -7.17683819e-05, 1.0, 8.42130893e-06, 1.0, 1.0, 1.0,
    };
    float std[num_of_rows][num_of_cols] = {
        1.49259451e-02, 3.22330006e-03, 1.96332955e-02, 1.96332663e-02, 8.86414915e-03, 1.26311744e+02, 1.40872237e+00, 1.98642367e-02, 2.82137393e-02, 1.65190416e+02, 3.03534892e-01, 1.14498188e-02, 2.04909340e+02, 1.20522046e+02, 5.97290116e-02, 4.98756114e-02, 1.37543738e-01, 1.08568109e+02, 1.89032465e-02, 6.36430634e-04, 1.0, 1.48271675e-04, 1.0, 1.0, 1.0,
    };
    //flat=((flat-mean)/std);
    for (int i = 0; i < num_of_rows; i++) {
        for (int j = 0; j < num_of_cols; j++) {
            flat[i][j] = (flat[i][j] - mean[i][j]) / std[i][j];
        }
    }
}



int main(int argc, const char* argv[]) {
 
  //Config:
  bool use_cuda=(argc>1)?true:false;
  if (use_cuda){std::cout<<"USING GPU"<<std::endl;}
  else{std::cout<<"USING CPU"<<std::endl;}

  //Get model:
  const char* model_path="../models/NN_weights_uniform_C.pt";
  torch::jit::script::Module model=get_model(model_path, use_cuda);

 // while (true){
      //Get data:
      float flat[num_of_rows][num_of_cols]={
          1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
      };

      //Preprocess data:
      preprocess_data(flat);

      //Forward run:
      at::Tensor output;
      output = predict(use_cuda, model, flat);

      //Convert tensor back to float, using vector:
      std::vector<float> out(output.data<float>(), output.data<float>() + output.numel());
      std::cout << out << '\n';
      
 // }
}


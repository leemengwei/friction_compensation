#define num_of_cols 25
#define num_of_rows 6
#define LOOP 1000
#define TIME_INTERVAL 24   //ms

#include <iostream>
#include <memory>
#include <sys/time.h>
#include <torch/script.h>
#include <pthread.h>
#include <unistd.h>
#include <string.h>

using namespace std;
struct thread_data{
  int id;
  bool use_cuda;
  torch::jit::script::Module model;
  float flat[num_of_rows][num_of_cols];
  at::Tensor output;
  };

//Tensor to float
std::vector<float> tensor_to_float(at::Tensor output){
  at::Tensor cpu_output;
  cpu_output = output.to(at::kCPU);
  std::vector<float> out(cpu_output.data<float>(), cpu_output.data<float>() + cpu_output.numel());
  return out;
}

//Function to load model
torch::jit::script::Module get_model(const char* model_path, bool use_cuda) {
  torch::jit::script::Module module;
  try {
    module = torch::jit::load(model_path);
    ;
  }
  catch (const c10::Error& e) {
    std::cerr << "Error loading the model\n";
    exit(-1);
  }
  cout<<"Model loaded:"<<model_path<<endl;
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
at::Tensor predict(bool use_cuda, torch::jit::script::Module model, float flat[][num_of_cols], bool speed_test) {
  //Convert
  std::vector<torch::jit::IValue> inputs = convert_data(use_cuda, flat);
  //and run
  at::Tensor output;
  output = model.forward(inputs).toTensor();
  //Speed test:
  if(speed_test){
    for (int i=0; i<1000; i++){ //warm-up
       output = model.forward(inputs).toTensor();
    }
    timeval starttime,endtime;
    gettimeofday(&starttime,0);
    for (int i=0; i<LOOP*10; i++){
       output = model.forward(inputs).toTensor();
    }
    gettimeofday(&endtime,0);
    double timeuse = 1000000*(endtime.tv_sec-starttime.tv_sec)+endtime.tv_usec-starttime.tv_usec; //us
    timeuse /= 1000; //ms
    timeuse /= LOOP*10; //peer buffer
    std::cout<<"Peer Buffer used:"<<timeuse<<"ms"<<std::endl;
  }
  else{
    ;
    }
  return output;
}

// 线程的运行函数
void* single_start(void*threadarg)
{
    //sleep(1.5);
    //cout << "Hello Runoob！" << endl;
    struct thread_data *my_data;
    my_data = (struct thread_data *) threadarg;
    my_data->output = predict(my_data->use_cuda, my_data->model, my_data->flat, false);
    //pthread_exit(NULL);
}

//Function to preprocess input data
void preprocess_data(float flat[][num_of_cols], float mean[][num_of_cols], float standard[][num_of_cols]) {
    //flat=((flat-mean)/standard);
    for (int i = 0; i < num_of_rows; i++) {
        for (int j = 0; j < num_of_cols; j++) {
            flat[i][j] = (flat[i][j] - mean[i][j]) / standard[i][j];
        }
    }
}

//Function to postprocess output data
void postprocess_data(std::vector<float> &out, float Y_mean, float Y_standard) {
    for (int i = 0; i < num_of_rows; i++) {
        out[i] = out[i]*Y_standard+Y_mean;
    }
}


int main(int argc, const char* argv[]) {
 
  //Config:
  bool speed_test=true;
  bool use_cuda=(argc>1)?true:false;
  if (use_cuda){std::cout<<"USING GPU"<<std::endl;}
  else{std::cout<<"USING CPU"<<std::endl;}

  //Get model:
  const char* model_path="../models/NN_weights_all_C.pt";
  torch::jit::script::Module model=get_model(model_path, use_cuda);

  //Get data:
  float flat[num_of_rows][num_of_cols]={
      1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
      1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
      1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
      1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
      1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
      1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
  };
  float mean[num_of_rows][num_of_cols] = {
        2.51065213e-04, 2.26194300e+02, -3.87426864e-03, -3.87426074e-03, 2.22841566e-04, 3.23421159e+01, -3.59547080e-01, -3.30278722e-03, -6.44383032e-04, 8.86972255e+01, 8.88570940e-02, 2.07215009e-03, -5.10639766e-02, -1.07646179e+00, -5.16060366e-03, -1.96551706e-05, -3.95824362e-04, -4.44822192e-01, 1.01560208e-03, -7.17683819e-05, 1.0, 8.42130893e-06, 1.0, 1.0, 1.0,
        2.51065213e-04, 2.26194300e+02, -3.87426864e-03, -3.87426074e-03, 2.22841566e-04, 3.23421159e+01, -3.59547080e-01, -3.30278722e-03, -6.44383032e-04, 8.86972255e+01, 8.88570940e-02, 2.07215009e-03, -5.10639766e-02, -1.07646179e+00, -5.16060366e-03, -1.96551706e-05, -3.95824362e-04, -4.44822192e-01, 1.01560208e-03, -7.17683819e-05, 1.0, 8.42130893e-06, 1.0, 1.0, 1.0,
        2.51065213e-04, 2.26194300e+02, -3.87426864e-03, -3.87426074e-03, 2.22841566e-04, 3.23421159e+01, -3.59547080e-01, -3.30278722e-03, -6.44383032e-04, 8.86972255e+01, 8.88570940e-02, 2.07215009e-03, -5.10639766e-02, -1.07646179e+00, -5.16060366e-03, -1.96551706e-05, -3.95824362e-04, -4.44822192e-01, 1.01560208e-03, -7.17683819e-05, 1.0, 8.42130893e-06, 1.0, 1.0, 1.0,
        2.51065213e-04, 2.26194300e+02, -3.87426864e-03, -3.87426074e-03, 2.22841566e-04, 3.23421159e+01, -3.59547080e-01, -3.30278722e-03, -6.44383032e-04, 8.86972255e+01, 8.88570940e-02, 2.07215009e-03, -5.10639766e-02, -1.07646179e+00, -5.16060366e-03, -1.96551706e-05, -3.95824362e-04, -4.44822192e-01, 1.01560208e-03, -7.17683819e-05, 1.0, 8.42130893e-06, 1.0, 1.0, 1.0,
        2.51065213e-04, 2.26194300e+02, -3.87426864e-03, -3.87426074e-03, 2.22841566e-04, 3.23421159e+01, -3.59547080e-01, -3.30278722e-03, -6.44383032e-04, 8.86972255e+01, 8.88570940e-02, 2.07215009e-03, -5.10639766e-02, -1.07646179e+00, -5.16060366e-03, -1.96551706e-05, -3.95824362e-04, -4.44822192e-01, 1.01560208e-03, -7.17683819e-05, 1.0, 8.42130893e-06, 1.0, 1.0, 1.0,
        2.51065213e-04, 2.26194300e+02, -3.87426864e-03, -3.87426074e-03, 2.22841566e-04, 3.23421159e+01, -3.59547080e-01, -3.30278722e-03, -6.44383032e-04, 8.86972255e+01, 8.88570940e-02, 2.07215009e-03, -5.10639766e-02, -1.07646179e+00, -5.16060366e-03, -1.96551706e-05, -3.95824362e-04, -4.44822192e-01, 1.01560208e-03, -7.17683819e-05, 1.0, 8.42130893e-06, 1.0, 1.0, 1.0,
   };
   float standard[num_of_rows][num_of_cols] = {
        1.49259451e-02, 3.22330006e-03, 1.96332955e-02, 1.96332663e-02, 8.86414915e-03, 1.26311744e+02, 1.40872237e+00, 1.98642367e-02, 2.82137393e-02, 1.65190416e+02, 3.03534892e-01, 1.14498188e-02, 2.04909340e+02, 1.20522046e+02, 5.97290116e-02, 4.98756114e-02, 1.37543738e-01, 1.08568109e+02, 1.89032465e-02, 6.36430634e-04, 1.0, 1.48271675e-04, 1.0, 1.0, 1.0,
        1.49259451e-02, 3.22330006e-03, 1.96332955e-02, 1.96332663e-02, 8.86414915e-03, 1.26311744e+02, 1.40872237e+00, 1.98642367e-02, 2.82137393e-02, 1.65190416e+02, 3.03534892e-01, 1.14498188e-02, 2.04909340e+02, 1.20522046e+02, 5.97290116e-02, 4.98756114e-02, 1.37543738e-01, 1.08568109e+02, 1.89032465e-02, 6.36430634e-04, 1.0, 1.48271675e-04, 1.0, 1.0, 1.0,
        1.49259451e-02, 3.22330006e-03, 1.96332955e-02, 1.96332663e-02, 8.86414915e-03, 1.26311744e+02, 1.40872237e+00, 1.98642367e-02, 2.82137393e-02, 1.65190416e+02, 3.03534892e-01, 1.14498188e-02, 2.04909340e+02, 1.20522046e+02, 5.97290116e-02, 4.98756114e-02, 1.37543738e-01, 1.08568109e+02, 1.89032465e-02, 6.36430634e-04, 1.0, 1.48271675e-04, 1.0, 1.0, 1.0,
        1.49259451e-02, 3.22330006e-03, 1.96332955e-02, 1.96332663e-02, 8.86414915e-03, 1.26311744e+02, 1.40872237e+00, 1.98642367e-02, 2.82137393e-02, 1.65190416e+02, 3.03534892e-01, 1.14498188e-02, 2.04909340e+02, 1.20522046e+02, 5.97290116e-02, 4.98756114e-02, 1.37543738e-01, 1.08568109e+02, 1.89032465e-02, 6.36430634e-04, 1.0, 1.48271675e-04, 1.0, 1.0, 1.0,
        1.49259451e-02, 3.22330006e-03, 1.96332955e-02, 1.96332663e-02, 8.86414915e-03, 1.26311744e+02, 1.40872237e+00, 1.98642367e-02, 2.82137393e-02, 1.65190416e+02, 3.03534892e-01, 1.14498188e-02, 2.04909340e+02, 1.20522046e+02, 5.97290116e-02, 4.98756114e-02, 1.37543738e-01, 1.08568109e+02, 1.89032465e-02, 6.36430634e-04, 1.0, 1.48271675e-04, 1.0, 1.0, 1.0,
        1.49259451e-02, 3.22330006e-03, 1.96332955e-02, 1.96332663e-02, 8.86414915e-03, 1.26311744e+02, 1.40872237e+00, 1.98642367e-02, 2.82137393e-02, 1.65190416e+02, 3.03534892e-01, 1.14498188e-02, 2.04909340e+02, 1.20522046e+02, 5.97290116e-02, 4.98756114e-02, 1.37543738e-01, 1.08568109e+02, 1.89032465e-02, 6.36430634e-04, 1.0, 1.48271675e-04, 1.0, 1.0, 1.0,
   };
  float Y_mean = -0.0014823869995695225;
  float Y_standard = 0.10306605601654062;

  timeval tic, toc;
  double tictoc;
  //Preprocess data:
  gettimeofday(&tic,0);
  preprocess_data(flat, mean, standard);
  gettimeofday(&toc,0);tictoc = 1000000*(toc.tv_sec-tic.tv_sec)+toc.tv_usec-tic.tv_usec;tictoc /= 1000;cout<<"pre"<<tictoc<<endl;

  //Forward run:
  at::Tensor output;
  output = predict(use_cuda, model, flat, speed_test);
  
  //Convert tensor back to float, using vector:
  std::vector<float> out;
  out = tensor_to_float(output);

  //Postprocess data:
  gettimeofday(&tic,0);
  postprocess_data(out, Y_mean, Y_standard);
  gettimeofday(&toc,0);tictoc = 1000000*(toc.tv_sec-tic.tv_sec)+toc.tv_usec-tic.tv_usec;tictoc /= 1000;cout<<"post"<<tictoc<<endl;
  std::cout << out << '\n';

  //Multi thread----------------------------:
  const char* model_path0="../models/NN_weights_all_C.pt";
  const char* model_path1="../models/NN_weights_all_C.pt";
  const char* model_path2="../models/NN_weights_all_C.pt";
  const char* model_path3="../models/NN_weights_all_C.pt";
  const char* model_path4="../models/NN_weights_all_C.pt";
  const char* model_path5="../models/NN_weights_all_C.pt";
  pthread_t t[6];
  struct thread_data td[6];
  td[0].id = 0;
  td[1].id = 1;
  td[2].id = 2;
  td[3].id = 3;
  td[4].id = 4;
  td[5].id = 5;
  td[0].use_cuda = use_cuda;
  td[1].use_cuda = use_cuda;
  td[2].use_cuda = use_cuda;
  td[3].use_cuda = use_cuda;
  td[4].use_cuda = use_cuda;
  td[5].use_cuda = use_cuda;
  td[0].model = get_model(model_path0, use_cuda);
  td[1].model = get_model(model_path1, use_cuda);
  td[2].model = get_model(model_path2, use_cuda);
  td[3].model = get_model(model_path3, use_cuda);
  td[4].model = get_model(model_path4, use_cuda);
  td[5].model = get_model(model_path5, use_cuda);
  memcpy(td[0].flat,flat,sizeof(flat));
  memcpy(td[1].flat,flat,sizeof(flat));
  memcpy(td[2].flat,flat,sizeof(flat));
  memcpy(td[3].flat,flat,sizeof(flat));
  memcpy(td[4].flat,flat,sizeof(flat));
  memcpy(td[5].flat,flat,sizeof(flat));

  float running_t = 0.0;
  for (int i = 0; i < LOOP; i++) {
    gettimeofday(&tic,0);
    //创建的线程id，线程属性，调用的函数，传入的参数
    pthread_create(&t[0], NULL, single_start, (void *)&td[0]);
    pthread_create(&t[1], NULL, single_start, (void *)&td[1]);
    pthread_create(&t[2], NULL, single_start, (void *)&td[2]);
    pthread_create(&t[3], NULL, single_start, (void *)&td[3]);
    pthread_create(&t[4], NULL, single_start, (void *)&td[4]);
    pthread_create(&t[5], NULL, single_start, (void *)&td[5]);
    //Join together
    pthread_join(t[0],NULL);
    pthread_join(t[1],NULL);
    pthread_join(t[2],NULL);
    pthread_join(t[3],NULL);
    pthread_join(t[4],NULL);
    pthread_join(t[5],NULL);
    //等线程退出后，进程才结束 否则强制了
    //pthread_exit(NULL);
    gettimeofday(&toc,0);
    tictoc = 1000000*(toc.tv_sec-tic.tv_sec)+toc.tv_usec-tic.tv_usec; //us
    tictoc /= 1000; //ms
    running_t += tictoc;
    std::cout << "All threads called " <<i<< '\n';

    //std::vector<float> out0;
    //out0 = tensor_to_float(td[0].output);
    //std::vector<float> out1;
    //out1 = tensor_to_float(td[1].output);
    //std::vector<float> out2;
    //out2 = tensor_to_float(td[2].output);
    //std::vector<float> out3;
    //out3 = tensor_to_float(td[3].output);
    //std::vector<float> out4;
    //out4 = tensor_to_float(td[4].output);
    //std::vector<float> out5;
    //out5 = tensor_to_float(td[5].output);
  
    ////Postprocess data:
    //postprocess_data(out0, Y_mean, Y_standard);
    //postprocess_data(out1, Y_mean, Y_standard);
    //postprocess_data(out2, Y_mean, Y_standard);
    //postprocess_data(out3, Y_mean, Y_standard);
    //postprocess_data(out4, Y_mean, Y_standard);
    //postprocess_data(out5, Y_mean, Y_standard);

    usleep(TIME_INTERVAL*1000);
  }
  std::cout<<"Multi thread Average used:"<<running_t/LOOP<<"ms over "<<LOOP<<" loops"<<std::endl;

  std::cout << "THE END" << '\n';
  
}


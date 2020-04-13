#define NUM_OF_COLS 5   //feature dims
#define NUM_OF_ROWS 1   //buffer length
#define LOOP 1000
#define TIME_INTERVAL 0   //ms
#define WARM_UP_TIME_SINGLE 100
#define WARM_UP_TIME_MULTI 100
#define REFRESH_CACHE false

#include <iostream>
#include <memory>
#include <time.h>
#include <torch/script.h>
//#include <pthread.h>
//#include <unistd.h>
#include<windows.h>
#include <string.h>
#include "ATen/Parallel.h"
#include <omp.h>
using namespace std;

struct thread_data{
  int id;
  bool use_cuda;
  torch::jit::script::Module model;
  float flat[NUM_OF_ROWS][NUM_OF_COLS];
  float X_mean[NUM_OF_COLS];
  float X_standard[NUM_OF_COLS];
  float Y_mean;
  float Y_standard;
  std::vector<float> final_out;
  };

//Function read txt data:
void read_X_stuff(string file, float X_stuff[])
{
  ifstream infile; 
  infile.open(file); 
  assert(infile.is_open()); 
  for(int this_col=0;this_col<NUM_OF_COLS;this_col++){
    infile>>X_stuff[this_col];
    }
  infile.close(); 
}

float read_Y_stuff(string file, float& Y_stuff)
{
  ifstream infile; 
  infile.open(file); 
  assert(infile.is_open()); 
  infile>>Y_stuff;
  infile.close(); 
}

//Function to preprocess input data
std::vector<torch::jit::IValue> preprocessing(float flat[][NUM_OF_COLS], float X_mean[], float X_standard[], float preprocessed_data[][NUM_OF_COLS], bool use_cuda) {
  //flat=((flat-mean)/standard);
  for (int i = 0; i < NUM_OF_ROWS; i++) {
      for (int j = 0; j < NUM_OF_COLS; j++) {
          preprocessed_data[i][j] = (flat[i][j] - X_mean[j]) / X_standard[j];
      }
  }
  //Float to tensor
  at::Tensor data = torch::from_blob(preprocessed_data, {NUM_OF_ROWS,NUM_OF_COLS}, torch::kFloat);
  data = data.toType(torch::kFloat);
  if (use_cuda){
    data=data.to(at::kCUDA);
   }
  std::vector<torch::jit::IValue> inputs;
  inputs.push_back(data);
  return inputs;
}

//Function to postprocess output data
std::vector<float> postprocessing(at::Tensor &output, float Y_mean, float Y_standard) {
  //Tensor to float
  at::Tensor cpu_output;
  cpu_output = output.to(at::kCPU);
  std::vector<float> out(cpu_output.data<float>(), cpu_output.data<float>() + cpu_output.numel());
  //flat=flat*standard+mean
  for (int i = 0; i < NUM_OF_ROWS; i++) {
      out[i] = out[i]*Y_standard+Y_mean;
  }
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

//Core to run
std::vector<float> predict(float flat[][NUM_OF_COLS], float X_mean[], float X_standard[], bool use_cuda, torch::jit::script::Module model, float Y_mean, float Y_standard, std::vector<float> final_out) {
  at::set_num_threads(1);
  float preprocessed_data[NUM_OF_ROWS][NUM_OF_COLS];
  std::vector<torch::jit::IValue> inputs;
  at::Tensor output;
  //Preprocess data:
  inputs = preprocessing(flat, X_mean, X_standard, preprocessed_data, use_cuda);
  //Forward:
  output = model.forward(inputs).toTensor();  
  //Postprocess data:
  final_out = postprocessing(output, Y_mean, Y_standard);
  return final_out;
}

//Core to dry-run
std::vector<float> predict_dry_run(float flat[][NUM_OF_COLS], float X_mean[], float X_standard[], bool use_cuda, torch::jit::script::Module model, float Y_mean, float Y_standard, std::vector<float> final_out) {
  //usleep(5*1000);//ms
  Sleep(5*1000);//ms
  return final_out;
}

void warm_up_single(const char* model_path, float flat[][NUM_OF_COLS], float X_mean[], float X_standard[], bool use_cuda, torch::jit::script::Module model, float Y_mean, float Y_standard, std::vector<float> final_out){
  cout<<"Warming up model..."<<model_path<<endl;
  for (int x=0;x<WARM_UP_TIME_SINGLE;x++){
    final_out = predict(flat, X_mean, X_standard, use_cuda, model, Y_mean, Y_standard, final_out);
  }
}

// 线程的运行函数
void* start_a_thread(void*threadarg){
  struct thread_data *my_data;
  my_data = (struct thread_data *) threadarg;
  for (int loop=0; loop<LOOP; loop++){
    if(REFRESH_CACHE){
      for(int row=0;row<NUM_OF_ROWS;row++){
        for(int col=0;col<NUM_OF_COLS;col++){
          my_data->flat[row][col] = loop;
        }
      }
    }
    //Warm up? So far, each thread is started only once and loop within, thus warming up itself. Yet this comsumption was undesirably calculated.
    my_data->final_out = predict(my_data->flat, my_data->X_mean, my_data->X_standard, my_data->use_cuda, my_data->model, my_data->Y_mean, my_data->Y_standard, my_data->final_out);
  //usleep(TIME_INTERVAL*1000);
  Sleep(TIME_INTERVAL*1000);
  }
  //pthread_exit(NULL);
}

/*
void threads_start_speed_test_all_in_one(bool use_cuda){  
  std::cout<<"\n\n\nNow on multi thread"<<endl;
  //Timers:
  timeval tic, toc;
  float tictoc;

  float flat1[NUM_OF_ROWS][NUM_OF_COLS]={
      0.0, 0.0, 0.0, 0.0, 0.0
      };
  float flat2[NUM_OF_ROWS][NUM_OF_COLS]={
      0.0, 0.0, 0.0, 0.0, 0.0
      };
  float flat3[NUM_OF_ROWS][NUM_OF_COLS]={
      0.0, 0.0, 0.0, 0.0, 0.0
      };
  float flat4[NUM_OF_ROWS][NUM_OF_COLS]={
      0.0, 0.0, 0.0, 0.0, 0.0
      };
  float flat5[NUM_OF_ROWS][NUM_OF_COLS]={
      0.0, 0.0, 0.0, 0.0, 0.0
      };
  float flat6[NUM_OF_ROWS][NUM_OF_COLS]={
      0.0, 0.0, 0.0, 0.0, 0.0
      };
  float X_mean1[NUM_OF_COLS];
  read_X_stuff("../statistics/X_mean_1", X_mean1);
  float X_mean2[NUM_OF_COLS];
  read_X_stuff("../statistics/X_mean_2", X_mean2);
  float X_mean3[NUM_OF_COLS];
  read_X_stuff("../statistics/X_mean_3", X_mean3);
  float X_mean4[NUM_OF_COLS];
  read_X_stuff("../statistics/X_mean_4", X_mean4);
  float X_mean5[NUM_OF_COLS];
  read_X_stuff("../statistics/X_mean_5", X_mean5);
  float X_mean6[NUM_OF_COLS];
  read_X_stuff("../statistics/X_mean_6", X_mean6);
  float X_standard1[NUM_OF_COLS];
  read_X_stuff("../statistics/X_std_1", X_standard1);
  float X_standard2[NUM_OF_COLS];
  read_X_stuff("../statistics/X_std_2", X_standard2);
  float X_standard3[NUM_OF_COLS];
  read_X_stuff("../statistics/X_std_3", X_standard3);
  float X_standard4[NUM_OF_COLS];
  read_X_stuff("../statistics/X_std_4", X_standard4);
  float X_standard5[NUM_OF_COLS];
  read_X_stuff("../statistics/X_std_5", X_standard5);
  float X_standard6[NUM_OF_COLS];
  read_X_stuff("../statistics/X_std_6", X_standard6);

  float Y_mean1;
  read_Y_stuff("../statistics/Y_mean_1", Y_mean1);
  float Y_mean2;
  read_Y_stuff("../statistics/Y_mean_2", Y_mean2);
  float Y_mean3;
  read_Y_stuff("../statistics/Y_mean_3", Y_mean3);
  float Y_mean4;
  read_Y_stuff("../statistics/Y_mean_4", Y_mean4);
  float Y_mean5;
  read_Y_stuff("../statistics/Y_mean_5", Y_mean5);
  float Y_mean6;
  read_Y_stuff("../statistics/Y_mean_6", Y_mean6);
  float Y_standard1;
  read_Y_stuff("../statistics/Y_std_1", Y_standard1);
  float Y_standard2;
  read_Y_stuff("../statistics/Y_std_2", Y_standard2);
  float Y_standard3;
  read_Y_stuff("../statistics/Y_std_3", Y_standard3);
  float Y_standard4;
  read_Y_stuff("../statistics/Y_std_4", Y_standard4);
  float Y_standard5;
  read_Y_stuff("../statistics/Y_std_5", Y_standard5);
  float Y_standard6;
  read_Y_stuff("../statistics/Y_std_6", Y_standard6);
  const char* model_path1="../models/NN_weights_all_C_1.pt";
  const char* model_path2="../models/NN_weights_all_C_2.pt";
  const char* model_path3="../models/NN_weights_all_C_3.pt";
  const char* model_path4="../models/NN_weights_all_C_4.pt";
  const char* model_path5="../models/NN_weights_all_C_5.pt";
  const char* model_path6="../models/NN_weights_all_C_6.pt";
  std::vector<float> final_out1;
  std::vector<float> final_out2;
  std::vector<float> final_out3;
  std::vector<float> final_out4;
  std::vector<float> final_out5;
  std::vector<float> final_out6;

  //Form td data struct
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
  td[0].model = get_model(model_path1, use_cuda);
  td[1].model = get_model(model_path2, use_cuda);
  td[2].model = get_model(model_path3, use_cuda);
  td[3].model = get_model(model_path4, use_cuda);
  td[4].model = get_model(model_path5, use_cuda);
  td[5].model = get_model(model_path6, use_cuda);
  td[0].Y_mean = Y_mean1;
  td[1].Y_mean = Y_mean2;
  td[2].Y_mean = Y_mean3;
  td[3].Y_mean = Y_mean4;
  td[4].Y_mean = Y_mean5;
  td[5].Y_mean = Y_mean6;
  td[0].Y_standard = Y_standard1;
  td[1].Y_standard = Y_standard2;
  td[2].Y_standard = Y_standard3;
  td[3].Y_standard = Y_standard4;
  td[4].Y_standard = Y_standard5;
  td[5].Y_standard = Y_standard6;
  memcpy(td[0].X_mean, X_mean1, sizeof(X_mean1));
  memcpy(td[1].X_mean, X_mean2, sizeof(X_mean2));
  memcpy(td[2].X_mean, X_mean3, sizeof(X_mean3));
  memcpy(td[3].X_mean, X_mean4, sizeof(X_mean4));
  memcpy(td[4].X_mean, X_mean5, sizeof(X_mean5));
  memcpy(td[5].X_mean, X_mean6, sizeof(X_mean6));
  memcpy(td[0].X_standard, X_standard1, sizeof(X_standard1));
  memcpy(td[1].X_standard, X_standard2, sizeof(X_standard2));
  memcpy(td[2].X_standard, X_standard3, sizeof(X_standard3));
  memcpy(td[3].X_standard, X_standard4, sizeof(X_standard4));
  memcpy(td[4].X_standard, X_standard5, sizeof(X_standard5));
  memcpy(td[5].X_standard, X_standard6, sizeof(X_standard6));
  memcpy(td[0].flat, flat1, sizeof(flat1));
  memcpy(td[1].flat, flat2, sizeof(flat2));
  memcpy(td[2].flat, flat3, sizeof(flat3));
  memcpy(td[3].flat, flat4, sizeof(flat4));
  memcpy(td[4].flat, flat5, sizeof(flat5));
  memcpy(td[5].flat, flat6, sizeof(flat6));
  td[0].final_out = final_out1;
  td[1].final_out = final_out2;
  td[2].final_out = final_out3;
  td[3].final_out = final_out4;
  td[4].final_out = final_out5;
  td[5].final_out = final_out6;

  //Starting threads:
  pthread_t t[6];
  float running_t = 0.0;
  //Warm up for each thread model:   <---- This is outside threads, thus actually useless.
  //for (int x=0;x<WARM_UP_TIME_MULTI;x++){
  //  for (int i=0;i<=5;i++){
  //    td[i].final_out = predict(td[i].flat, td[i].X_mean, td[i].X_standard, td[i].use_cuda, td[i].model, td[i].Y_mean, td[i].Y_standard, td[i].final_out);
  //  }
  //}
  gettimeofday(&tic,0);
  //创建的线程id，线程属性，调用的函数，传入的参数
  pthread_create(&t[0], NULL, start_a_thread, (void *)&td[0]);
  pthread_create(&t[1], NULL, start_a_thread, (void *)&td[1]);
  pthread_create(&t[2], NULL, start_a_thread, (void *)&td[2]);
  pthread_create(&t[3], NULL, start_a_thread, (void *)&td[3]);
  pthread_create(&t[4], NULL, start_a_thread, (void *)&td[4]);
  pthread_create(&t[5], NULL, start_a_thread, (void *)&td[5]);
  //Join together
  pthread_join(t[0],NULL);
  pthread_join(t[1],NULL);
  pthread_join(t[2],NULL);
  pthread_join(t[3],NULL);
  pthread_join(t[4],NULL);
  pthread_join(t[5],NULL);
  //等线程退出后，进程才结束 否则强制
  //pthread_exit(NULL);
  //running_t += tictoc;
  gettimeofday(&toc,0);
  tictoc = 1000000*(toc.tv_sec-tic.tv_sec)+toc.tv_usec-tic.tv_usec; //us
  tictoc /= 1000; //ms
  std::cout<<"\nMulti thread time used over each LOOP("<<LOOP<<"): "<<tictoc/LOOP<<"ms"<<std::endl;

  //Report:
  std::cout << "Output" << '\n';
  for (int i=0;i<=5;i++){
    for (int j=0;j<NUM_OF_ROWS;j++){
      std::cout<<td[i].final_out[j]<<" ";
    }
    std::cout<<endl;
  }
}
*/

/*
void single_speed_test_loop(float flat[][NUM_OF_COLS], float X_mean[], float X_standard[], bool use_cuda, torch::jit::script::Module model, float Y_mean, float Y_standard, std::vector<float> final_out){
  //Now speed test run:
  timeval tic, toc;
  float tictoc;
  gettimeofday(&tic,0);
  for (int loop=0;loop<LOOP;loop++){
    if(REFRESH_CACHE){
      for(int row=0;row<NUM_OF_ROWS;row++){
        for(int col=0;col<NUM_OF_COLS;col++){
          flat[row][col] = loop;
        }
      }
    }
    //Forward run:
    final_out = predict(flat, X_mean, X_standard, use_cuda, model, Y_mean, Y_standard, final_out);
    usleep(TIME_INTERVAL*1000);
  }
  gettimeofday(&toc,0);
  tictoc = 1000000*(toc.tv_sec-tic.tv_sec)+toc.tv_usec-tic.tv_usec; //us
  tictoc /= 1000; //ms
  std::cout<<"Time used over LOOPs("<<LOOP<<"): "<<tictoc/LOOP<<"ms"<<endl;

  //Report:
  std::cout << "Output" << '\n';
  for (int i=0;i<NUM_OF_ROWS;i++){
    std::cout<<final_out[i]<<" ";
  }
}
*/

int main(int argc, const char* argv[]) {

  //Report status:
  bool use_cuda=(argc>1)?true:false;
  if (use_cuda){std::cout<<"USING GPU"<<std::endl;}
  else{std::cout<<"USING CPU"<<std::endl;}
  std::cout<<"Warmed up for (s)"<<WARM_UP_TIME_SINGLE<<" and (m)"<<WARM_UP_TIME_MULTI<<" times. Refresh cache: "<<REFRESH_CACHE<<endl;
  
  //Configurations:
  const char* model_path = "../models/NN_weights_all_C_1.pt";
  const char* X_mean_filename = "../statistics/X_mean_1";
  const char* X_std_filename = "../statistics/X_std_1";
  const char* Y_mean_filename = "../statistics/Y_mean_1";
  const char* Y_std_filename = "../statistics/Y_std_1";
 
  //Get model:
  torch::jit::script::Module model=get_model(model_path, use_cuda);

  //Get data:
  float flat[NUM_OF_ROWS][NUM_OF_COLS] = {0.0, 0.0, 0.0, 0.0, 0.0};
  float X_mean[NUM_OF_COLS];
  float X_standard[NUM_OF_COLS]; 
  float Y_mean;
  float Y_standard;
  read_X_stuff(X_mean_filename, X_mean);
  read_X_stuff(X_std_filename, X_standard);
  read_Y_stuff(Y_mean_filename, Y_mean);
  read_Y_stuff(Y_std_filename, Y_standard);
  std::vector<float> final_out;

  //Warm up single:
  warm_up_single(model_path, flat, X_mean, X_standard, use_cuda, model, Y_mean, Y_standard, final_out);

  //Run once API:
  final_out = predict(flat, X_mean, X_standard, use_cuda, model, Y_mean, Y_standard, final_out);

  //Report:
  std::cout << "Output" << '\n';
  for (int i=0;i<NUM_OF_ROWS;i++){
    std::cout<<final_out[i]<<" ";
  }
  std::cout << endl;

  //---------------------------Single thread speed test---------------------:
  //single_speed_test_loop(flat, X_mean, X_standard, use_cuda, model, Y_mean, Y_standard, final_out);

  //---------------------------Multi_threads speed test---------------------:
  //threads_start_speed_test_all_in_one(use_cuda);

}


//END



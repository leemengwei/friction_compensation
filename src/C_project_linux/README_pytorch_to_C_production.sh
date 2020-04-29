
#CMAKE Instructions:
#假定已经有通过pytorch jit得到的C版本的pt模型，
#如下做法，搭建C环境加载、运行模型。（纯净环境，无gpu和caffe2需求，只是需要安装cuda和cudnn的库依赖(对于有显卡的来说，没有的更简单)）
#参考：https://pytorch.org/tutorials/advanced/cpp_export.html

#From scratch build:
#ExternalPrerequest:
#cmake
#g++
#build-essential     (g++)



1. Install cuda
  	-Sh cuda-xxxx.run
 
2. Install cudnn
	-Copy cudnn files to cuda path (perhaps /usr/local/cuda-9/0/)
	-sudo dpkg -i libcudnn7_7.0.3.11-1+cuda9.0_amd64.deb
	-sudo dpkg -i libcudnn7-dev_7.0.3.11-1+cuda9.0_amd64.deb
	-sudo dpkg -i libcudnn7-doc_7.0.3.11-1+cuda9.0_amd64.deb

	-Copy cudnn files to cuda path (perhaps /usr/local/cuda-10/0/)
        -sudo dpkg -i libcudnn7_7.6.5.32-1+cuda10.2_amd64.deb
        -sudo dpkg -i libcudnn7-dev_7.6.5.32-1+cuda10.2_amd64.deb
        -sudo dpkg -i libcudnn7-doc_7.6.5.32-1+cuda10.2_amd64.deb


3. Exec compile with libtorch_gpu
        mkdir build;
        rm build/* -r;cd build;cmake ..;cd ..;
	cd build/;make;cd ..;

4. Ok to run
	cd build/;./C_load_and_run_API;cd ..; #run single thread and multi thread
	./run_all_cpu.sh;   #multi process cpu (checkout first part given by single thread by C++)
	./run_all_gpu.sh;   #multi process gpu (checkout first part given by single thread by C++)

#MAKE Instructions:
#https://www.cnblogs.com/lidabo/p/6381772.html
#By g++: 
#When put all stuff in one cpp:
export LD_LIBRARY_PATH=./libtorch_cpu/lib/
g++ C_load_and_run_API.cpp -Ilibtorch_cpu/include -Llibtorch_cpu/lib/ -ltorch -lc10 -lgomp --std=c++11

#When compile from seperate static libraries:
export LD_LIBRARY_PATH=./libtorch_cpu/lib/
export LD_LIBRARY_PATH=./:$LD_LIBRARY_PATH
#seperate compile:
g++ -c get_model.cpp -Ilibtorch_cpu/include -Llibtorch_cpu/lib/ -ltorch -lc10 -lgomp --std=c++11
g++ -c get_data.cpp -Ilibtorch_cpu/include -Llibtorch_cpu/lib/ -ltorch -lc10 -lgomp --std=c++11
g++ -c predict.cpp -Ilibtorch_cpu/include -Llibtorch_cpu/lib/ -ltorch -lc10 -lgomp --std=c++11
#generate static libs:
ar -cr libpredict.a predict.o
ar -cr libget_model.a get_model.o
ar -cr libget_data.a get_data.o
#Static link:
g++ -o main C_load_and_run_API.cpp -Ilibtorch_cpu/include -Llibtorch_cpu/lib/ -ltorch -lc10 -lgomp -L./ -lget_model -lget_data -lpredict --std=c++11





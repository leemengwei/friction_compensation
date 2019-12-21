#假定已经有通过pytorch jit得到的C版本的pt模型，
#如下做法，搭建C环境加载、运行模型。（纯净环境，无gpu和caffe2需求，只是需要安装cuda和cudnn的库依赖(对于有显卡的来说，没有的更简单)）
#参考：https://pytorch.org/tutorials/advanced/cpp_export.html

#Instructions:
1. Install cuda
  	-Sh cuda-xxxx.run
 
2. Install cudnn
	-Copy cudnn files to cuda path (perhaps /usr/local/cuda-9/0/)
	-sudo dpkg -i libcudnn7_7.0.3.11-1+cuda9.0_amd64.deb
	-sudo dpkg -i libcudnn7-dev_7.0.3.11-1+cuda9.0_amd64.deb
	-sudo dpkg -i libcudnn7-doc_7.0.3.11-1+cuda9.0_amd64.deb

3. Exec compile with libtorch
        mkdir build; cd build
	-cmake -DCMAKE_PREFIX_PATH=/home/user/lee/libtorch ..
	and
	-make

4. Ok to run

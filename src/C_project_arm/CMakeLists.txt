cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(custom_ops)

set(name "C_load_and_run_API")
set(CMAKE_CXX_STANDARD 11)
#set(CMAKE_PREFIX_PATH "/home/nvidia/limengwei/arm_friction_src/libtorch_cuda")
#set(CMAKE_PREFIX_PATH "/home/nvidia/liuxu/pytorch/build/")
#set(CMAKE_PREFIX_PATH "/home/nvidia/liuxu/pytorch/torch/")
#set(CMAKE_PREFIX_PATH "/home/nvidia/liuxu/pytorch/build/lib.linux-aarch64-3.5/torch/")
#set(CMAKE_PREFIX_PATH "/home/nvidia/limengwei/pytorch1.1.0/build/")
#set(Torch_DIR /home/nvidia/limengwei/pytorch1.1.0//torch/share/cmake/Torch) 
#set(CMAKE_PREFIX_PATH "/home/nvidia/limengwei/pytorch1.1.0/build/lib.linux-aarch64-3.5/torch")
set(CMAKE_PREFIX_PATH "/home/nvidia/limengwei/pytorch1.3.1/build/lib.linux-aarch64-3.5/torch")

find_package(Torch REQUIRED)

add_executable(${name} ${name}.cpp)
target_link_libraries(${name} "${TORCH_LIBRARIES}")

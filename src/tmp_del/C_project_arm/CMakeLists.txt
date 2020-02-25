cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(custom_ops)

set(name "C_load_and_run_API")
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_PREFIX_PATH "/home/nvidia/limengwei/C_project_arm/libtorch_cuda_pre")

find_package(Torch REQUIRED)

add_executable(${name} ${name}.cpp)
target_link_libraries(${name} "${TORCH_LIBRARIES}")

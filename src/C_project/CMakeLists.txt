cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(custom_ops)

find_package(Torch REQUIRED)

add_executable(C_load_and_run C_load_and_run.cpp)
target_link_libraries(C_load_and_run "${TORCH_LIBRARIES}")
set_property(TARGET C_load_and_run PROPERTY CXX_STANDARD 11)

cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(custom_ops)

find_package(Torch REQUIRED)

#set(name "C_load_and_run")
set(name "C_load_and_run_API")

add_executable(${name} ${name}.cpp)
target_link_libraries(${name} "${TORCH_LIBRARIES}")
set_property(TARGET ${name} PROPERTY CXX_STANDARD 11)

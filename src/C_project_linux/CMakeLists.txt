cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(custom_ops)

set(name "C_load_and_run_API")
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_PREFIX_PATH "../C_project_setups/libtorch_cuda")

find_package(Torch REQUIRED)

add_executable(${name} ${name}.cpp)
target_link_libraries(${name} "${TORCH_LIBRARIES}")
#set_property(TARGET ${name} PROPERTY CXX_STANDARD 11)

# According to https://github.com/pytorch/pytorch/issues/25457,
# the DLLs need to be copied to avoid memory errors.
if (MSVC)
  file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
  add_custom_command(TARGET ${name}
                     POST_BUILD
                     COMMAND ${CMAKE_COMMAND} -E copy_if_different
                     ${TORCH_DLLS}
                     $<TARGET_FILE_DIR:${name}>)
endif (MSVC)

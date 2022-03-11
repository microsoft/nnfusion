cmake_minimum_required(VERSION 3.18)

project(memopt C CXX CUDA)
list(APPEND CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake/Modules)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(CUDAToolkit REQUIRED)
find_package(CUDNN 7.5 REQUIRED)
aux_source_directory(src conv_source)
add_library(simple_conv SHARED ${conv_source})
target_link_libraries(simple_conv CUDA::cudart ${CUDNN_LIBRARY_PATH})
target_include_directories(simple_conv PUBLIC ${CUDNN_INCLUDE_PATH})
set_property(TARGET simple_conv PROPERTY CUDA_ARCHITECTURES 61)

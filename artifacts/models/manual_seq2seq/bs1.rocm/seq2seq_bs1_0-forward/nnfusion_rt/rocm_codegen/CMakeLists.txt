project(main_test)
cmake_minimum_required(VERSION 3.5)

SET(SRC "nnfusion_rt.cpp" CACHE STRING "codegen source file")
SET(TARGET_NAME "nnfusion_naive_rt" CACHE STRING "codegen target name")

set(CMAKE_CXX_COMPILER /opt/rocm/bin/hipcc)
set(CMAKE_CXX_FLAGS "-O2 -Wno-ignored-attributes -Wno-duplicate-decl-specifier -save-temps --amdgpu-target=gfx908")
add_library(${TARGET_NAME} SHARED ${SRC})

include_directories(
    /opt/rocm/include
    /opt/rocm/rocblas/include
    /opt/rocm/rocrand/include
    /opt/rocm/hiprand/include
    /opt/rocm/hipsparse/include)

target_link_libraries(${TARGET_NAME} /opt/rocm/lib/libMIOpen.so /opt/rocm/lib/librocblas.so) 

add_executable(main_test main_test.cpp)
target_link_libraries(main_test ${TARGET_NAME}) 
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.


// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "codegen_langunit.hpp"

LU_DEFINE(nnfusion::codegen::cmake::cblas,
          R"(
if (NOT TARGET libmkl)
include(mkl/mkl.cmake)
endif()
target_link_libraries(nnfusion_cpu_rt pthread libmkl)
)");

LU_DEFINE(nnfusion::codegen::cmake::eigen,
          R"(
if (NOT TARGET eigen)
include(eigen/eigen.cmake)
endif()
target_link_libraries(${TARGET_NAME} eigen)
)");

LU_DEFINE(nnfusion::codegen::cmake::mlas,
          R"(
if (NOT TARGET mlas)
include(mlas/mlas.cmake)
endif()
target_link_libraries(${TARGET_NAME} mlas)
)");

LU_DEFINE(nnfusion::codegen::cmake::threadpool,
          R"(
if (NOT TARGET threadpool)
include(threadpool/threadpool.cmake)
endif()
target_link_libraries(${TARGET_NAME} threadpool)
)");

LU_DEFINE(nnfusion::codegen::cmake::threads,
          R"(
find_package(Threads REQUIRED)
target_link_libraries(${TARGET_NAME} Threads::Threads)
)");

LU_DEFINE(nnfusion::codegen::cmake::super_scaler,
          R"(
find_package(MPI)
include_directories(${MPI_INCLUDE_PATH})
find_library(SUPER_SCALER_LIBRARIES libsuper_scaler.so ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(${TARGET_NAME} 
    ${MPI_LIBRARIES}
    ${SUPER_SCALER_LIBRARIES}
    nccl)   
)");

LU_DEFINE(nnfusion::codegen::cmake::rocm_super_scaler,
          R"(
find_package(MPI)
include_directories(${MPI_INCLUDE_PATH})
find_library(ssrocm libsuper_scaler_rocm.so ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(${TARGET_NAME}
    ${MPI_LIBRARIES}
    ${ssrocm}
)");

LU_DEFINE(nnfusion::codegen::cmake::cuda_lib,
          R"(
link_directories(${CUDA_TOOLKIT_ROOT_DIR}/lib64)

find_path(CUDNN_INCLUDE_DIR cudnn.h
    HINTS ${CUDA_TOOLKIT_ROOT_DIR}
    PATH_SUFFIXES cuda/include include)

include_directories(${CUDNN_INCLUDE_DIR} ${CUDA_INCLUDE_DIRS})

find_library(CUDNN_LIBRARY cudnn
    HINTS ${CUDA_TOOLKIT_ROOT_DIR}
    PATH_SUFFIXES lib lib64 cuda/lib cuda/lib64 lib/x64)

find_library(CUDA_cuda_LIBRARY cuda ${CUDA_TOOLKIT_ROOT_DIR}/lib64/stubs)
find_library(CUDA_cudart_LIBRARY libcudart.so ${CUDA_TOOLKIT_ROOT_DIR}/lib64)

target_link_libraries(${TARGET_NAME}
    ${CUDA_cuda_LIBRARY}
    ${CUDA_cudart_LIBRARY}
    ${CUDA_LIBRARIES}
    ${CUDA_CUBLAS_LIBRARIES}
    ${CUDNN_LIBRARY})
)");

LU_DEFINE(nnfusion::codegen::cmake::rocm_lib,
          R"(
include_directories(
    /opt/rocm/include
    /opt/rocm/rocblas/include
    /opt/rocm/rocrand/include
    /opt/rocm/hiprand/include
    /opt/rocm/hipsparse/include)

target_link_libraries(${TARGET_NAME} /opt/rocm/lib/libMIOpen.so /opt/rocm/lib/librocblas.so) 
)");

LU_DEFINE(nnfusion::codegen::helper::debug,
          R"(

inline void Debug(std::string name, float* tensor_ptr, std::string inputs, size_t debug_size = 10, size_t offset=0, bool check_finite=true)
{
    float* host_tensor = (float*)malloc(sizeof(float) * debug_size);
    CUDA_SAFE_CALL(cudaDeviceSynchronize());
    CUDA_SAFE_CALL(cudaMemcpy(host_tensor, tensor_ptr + offset,  sizeof(float) * debug_size, cudaMemcpyDeviceToHost));
    CUDA_SAFE_CALL(cudaDeviceSynchronize());
    double sum = 0.0;
    for (size_t i = 0; i < debug_size; i++) sum += host_tensor[i];
    size_t print_size = min((size_t)10, debug_size);
    printf("%s: ", name.c_str());
    printf("sum=%e; ", sum);
    for (int i = 0; i < print_size; ++i) printf("%e ", host_tensor[i]);
    printf("...(size= %lu end with %e ) :", debug_size, host_tensor[debug_size - 1]);
    //print with an offset
    size_t print_offset = debug_size / 3;
    print_size = min((size_t)10, debug_size - print_offset);
    for (int i = 0; i < print_size; ++i) printf("%e ", host_tensor[i + print_offset]);
    printf("...(offset= %lu) ", print_offset);
    printf(": %s\n", inputs.c_str());
    if (check_finite)
        for (size_t ii = 0; ii < debug_size; ii++)
            if (!isfinite(host_tensor[ii]))
            {
                printf("Infinite found at %s[%lu]=%e\n", name.c_str(), ii, host_tensor[ii]);
                exit(1);
            }
    free(host_tensor);
}
)");

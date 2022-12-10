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

LU_DEFINE(nnfusion::codegen::cmake::superscaler_cuda,
          R"(
if (NOT TARGET superscaler)
set(TARGET_GPU_PLATFORM "CUDA" CACHE STRING "Choose your GPU platform: CUDA or ROCm")
include(superscaler/superscaler.cmake)
endif()
target_link_libraries(${TARGET_NAME} superscaler)
)");

LU_DEFINE(nnfusion::codegen::cmake::superscaler_rocm,
          R"(
if (NOT TARGET superscaler)
set(TARGET_GPU_PLATFORM "ROCm" CACHE STRING "Choose your GPU platform: CUDA or ROCm")
include(superscaler/superscaler.cmake)
endif()
target_link_libraries(${TARGET_NAME} superscaler)
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

LU_DEFINE(nnfusion::codegen::cmake::cub, R"(
if (NOT TARGET CUB)
include(cub/cub.cmake)
endif()
add_dependencies(${TARGET_NAME} CUB)
include_directories(${CUB_INCLUDE_DIR})
)");

LU_DEFINE(nnfusion::codegen::cmake::cutlass, R"(
include_directories(cutlass/include/)
include_directories(cutlass/examples/42_fused_multi_head_attention)
include_directories(cutlass/tools/util/include/)
)");

LU_DEFINE(nnfusion::codegen::helper::cuda_half_debug,
          R"(
extern "C" __global__ void Convert_half_float0(half* input0, float* output0, int bound)
{
    if (bound != 0 && threadIdx.x >= bound)
        return;
    output0[threadIdx.x] = (float)(input0[threadIdx.x]);

}

extern "C" __global__ void Convert_half_float1(half* input0, float* output0, int blks, int bound)
{
    if (bound !=0 && blockIdx.x * blks + threadIdx.x >= bound)
        return;
    output0[blockIdx.x * blks + threadIdx.x] = (float)(input0[blockIdx.x * blks + threadIdx.x]);

}

extern void Convert_half_float_Call0(const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream, half* input0, float* output0, int bound) {
    Convert_half_float0<<<grids, blocks, mem, stream>>>(input0, output0, bound);
}

extern void Convert_half_float_Call1(const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream, half* input0, float* output0, int blks, int bound) {
    Convert_half_float1<<<grids, blocks, mem, stream>>>(input0, output0, blks, bound);
}

)")

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
    printf("%s: \n", name.c_str());
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

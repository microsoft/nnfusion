# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.


set(SRC
    cuda_emitter.cpp
    cuda_cudnn.cpp
    cuda_helper.cpp
    cuda_langunit.cpp
    cuda_testutil.cpp
    util/gpu_util.cpp
)

file(GLOB wild_cuda_kernels kernels/*.cpp)
file(GLOB wild_rocm_kernels kernels/rocm/*.cpp)

set(KERNELS
    ${wild_cuda_kernels}
    ${wild_rocm_kernels}
)

add_library(kernels_cuda STATIC ${SRC} ${KERNELS})
target_include_directories(kernels_cuda SYSTEM PUBLIC
    ${GLOBAL_INCLUDE_PATH}
)
target_compile_options(kernels_cuda PRIVATE "-fPIC")

target_link_libraries(kernels_cuda PRIVATE
    kernels_registration gflags ${CURL_LIBRARIES}
)

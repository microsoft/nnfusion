# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.


set(SRC
    cpu_kernel_emitter.cpp
    cpu_langunit.cpp
    cpu_helper.cpp
    barrier.cpp
)

file(GLOB eigen_kernels eigen/*.cpp)
file(GLOB mkl_kernels mkl/*.cpp)
file(GLOB reference_kernels reference/*.cpp)
file(GLOB general_kernels general/*.cpp)
file(GLOB mlas_kernels mlas/*.cpp)
file(GLOB antares_kernels antares/*.cpp)
file(GLOB simd_kernels simd/*.cpp)

set(KERNELS
    ${reference_kernels}
    ${general_kernels}
    ${mkl_kernels}
    ${eigen_kernels}
    ${simd_kernels}
    ${mlas_kernels}
)

add_library(kernels_cpu STATIC ${SRC} ${KERNELS})
target_include_directories(kernels_cpu SYSTEM PUBLIC
    ${GLOBAL_INCLUDE_PATH}
)
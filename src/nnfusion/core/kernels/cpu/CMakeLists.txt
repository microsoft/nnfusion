# Microsoft (c) 2019, NNFusion Team

set(SRC
    cpu_kernel_emitter.cpp
    cpu_langunit.cpp
    barrier.cpp
)

file(GLOB eigen_kernels eigen/*.cpp)
file(GLOB mkl_kernels mkl/*.cpp)
file(GLOB reference_kernels reference/*.cpp)
file(GLOB general_kernels general/*.cpp)
file(GLOB mlas_kernels mlas/*.cpp)

set(KERNELS
    ${eigen_kernels}
    ${mkl_kernels}
    ${reference_kernels}
    ${general_kernels}
    ${mlas_kernels}
)

add_library(kernels_cpu STATIC ${SRC} ${KERNELS})
target_include_directories(kernels_cpu SYSTEM PUBLIC
    ${NGRAPH_INCLUDE_PATH}
)
target_compile_options(kernels_cpu PRIVATE "-fPIC")

target_link_libraries(kernels_cpu  PRIVATE
    kernels_registration
)

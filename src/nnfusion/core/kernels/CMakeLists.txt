# Microsoft (c) 2019, NNFusion Team

#-------lib_kernels_registration----------
set(SRC
    kernel_emitter.cpp
    kernel_registration.cpp
    common_langunit.cpp
)
add_library(kernels_registration STATIC ${SRC})
target_include_directories(kernels_registration SYSTEM PUBLIC
    ${NGRAPH_INCLUDE_PATH}
)
target_link_libraries(kernels_registration PRIVATE
    nnfusion_common nnfusion_util nnfusion_graph nnfusion_operators
)
target_compile_options(kernels_registration PRIVATE "-fPIC")

add_subdirectory(cuda_gpu)
add_subdirectory(cpu)

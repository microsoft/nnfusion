# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.


#-------lib_kernels_registration----------
set(SRC
    kernel_emitter.cpp
    kernel_registration.cpp
    common_langunit.cpp
    antares_ke_imp.cpp
)
add_library(kernels_registration STATIC ${SRC})
target_include_directories(kernels_registration SYSTEM PUBLIC
    ${GLOBAL_INCLUDE_PATH}
)
target_link_libraries(kernels_registration PRIVATE
    nnfusion_common nnfusion_graph nnfusion_operators ${CURL_LIBRARIES}
)
target_compile_options(kernels_registration PRIVATE "-fPIC")

add_subdirectory(cuda_gpu)
add_subdirectory(cpu)
add_subdirectory(hlsl)
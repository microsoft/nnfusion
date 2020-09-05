# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.


set(SRC
    blockcuda.cpp
)

add_library(kernels_cache STATIC ${SRC})
target_include_directories(kernels_cache SYSTEM PUBLIC
    ${GLOBAL_INCLUDE_PATH}
)
target_compile_options(kernels_cache PRIVATE "-fPIC")

target_link_libraries(kernels_cache PRIVATE
    kernels_registration gflags
)

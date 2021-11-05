# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.


set(SRC
    hlsl_kernel_emitter.cpp
    constant.cpp
    parameter.cpp
    result.cpp
    hlsl_langunit.cpp
)

add_library(kernels_hlsl STATIC ${SRC})
target_include_directories(kernels_hlsl SYSTEM PUBLIC
    ${NGRAPH_INCLUDE_PATH}
)
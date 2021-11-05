# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.


set(SRC
    cuda.cpp
    graphcore.cpp
    hlsl.cpp
    rocm.cpp
    cpu.cpp
    reversed_dfs_visitor.cpp
    degree_based_visitor.cpp
)
add_library(nnfusion_engine_device STATIC ${SRC})
target_include_directories(nnfusion_engine_device SYSTEM PUBLIC
    ${GLOBAL_INCLUDE_PATH}
)

target_link_libraries(nnfusion_engine_device
    nnfusion_engine_base
    nnfusion_ir
    nnfusion_engine_pass
)
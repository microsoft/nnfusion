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

target_link_libraries(nnfusion_engine_device PRIVATE
    nnfusion_graph ${CURL_LIBRARIES}
)
target_compile_options(nnfusion_engine_device PRIVATE "-fPIC")
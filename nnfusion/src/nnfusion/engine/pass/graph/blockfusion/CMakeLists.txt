# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.


set(SRC
    block_kernel_schedule.cpp
    block_parallel_device.cpp
    blockfusion_codegen.cpp
    blockfusion_profiler.cpp
    blockfusion_optimizer.cpp
)

add_library(nnfusion_engine_pass_graph_blockfusion STATIC ${SRC})
target_include_directories(nnfusion_engine_pass_graph_blockfusion SYSTEM PUBLIC
    ${GLOBAL_INCLUDE_PATH}
)

target_link_libraries(nnfusion_engine_pass_graph_blockfusion PRIVATE nnfusion_common
gflags
)

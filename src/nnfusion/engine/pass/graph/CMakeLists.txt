# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.


add_subdirectory(blockfusion)

set(SRC
    assign_layout_pass.cpp
    kernel_fusion_pass.cpp
    gemm_fusion_pass.cpp
    op_inplace_pass.cpp
    graph_pass.cpp
    gradient_weight_mapping_pass.cpp
    gnode_device_dispatcher.cpp
    kernel_selection.cpp
    blockfusion_pass.cpp
    assign_async_info_pass.cpp
    kernel_profiling_pass.cpp
    runtime_const_folding_pass.cpp
    pattern_substitution.cpp
    batchnorm_inference_folding_pass.cpp
    dot_transpose_pass.cpp
)

add_library(nnfusion_engine_pass_graph STATIC ${SRC})
target_include_directories(nnfusion_engine_pass_graph SYSTEM PUBLIC
    ${GLOBAL_INCLUDE_PATH}
)
target_compile_options(nnfusion_engine_pass_graph PRIVATE "-fPIC")

target_link_libraries(nnfusion_engine_pass_graph PRIVATE nnfusion_common nnfusion_engine_pass_graph_blockfusion ngraph_engine_pass_graph
gflags ${CURL_LIBRARIES}
)

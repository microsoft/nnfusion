# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.


add_subdirectory(blockfusion)
add_subdirectory(autodiff)
add_subdirectory(bertfusion_optimizer)

set(SRC
    assign_layout_pass.cpp
    kernel_fusion_pass.cpp
    gemm_fusion_pass.cpp
    op_inplace_pass.cpp
    graph_pass.cpp
    gradient_weight_mapping_pass.cpp
    gnode_device_dispatcher.cpp
    kernel_tuning.cpp
    kernel_selection.cpp
    blockfusion_pass.cpp
    bertfusion_pass.cpp
    assign_async_info_pass.cpp
    kernel_profiling_pass.cpp
    runtime_const_folding_pass.cpp
    common_subexpression_elimination_pass.cpp
    pattern_substitution.cpp
    batchnorm_inference_folding_pass.cpp
    autodiff_pass.cpp
    dot_transpose_pass.cpp
    reduce_fusion_pass.cpp
    superscaler_dataparallelism_pass.cpp
    hlsl_required_pass.cpp
)

add_library(nnfusion_engine_pass_graph STATIC ${SRC})
target_include_directories(nnfusion_engine_pass_graph SYSTEM PUBLIC
    ${GLOBAL_INCLUDE_PATH}
)
target_compile_options(nnfusion_engine_pass_graph PRIVATE "-fPIC")

target_link_libraries(nnfusion_engine_pass_graph PRIVATE nnfusion_common nnfusion_engine_pass_graph_blockfusion nnfusion_engine_pass_graph_bertfusion
gflags ${CURL_LIBRARIES} ${WholeArchiveFlag} nnfusion_engine_pass_graph_autodiff ${NoWholeArchiveFlag}
)

# Microsoft (c) 2019, NNFusion Team

set(SRC
    assign_layout_pass.cpp
    kernel_fusion_pass.cpp
    gemm_fusion_pass.cpp
    op_inplace_pass.cpp
    graph_pass.cpp
    manager.cpp
    gradient_weight_mapping_pass.cpp
    device_dispatcher.cpp
    kernel_selection.cpp
    assign_async_info_pass.cpp
)

add_library(nnfusion_engine_pass_graph STATIC ${SRC})
target_include_directories(nnfusion_engine_pass_graph SYSTEM PUBLIC
    ${NGRAPH_INCLUDE_PATH}
)
target_compile_options(nnfusion_engine_pass_graph PRIVATE "-fPIC")

target_link_libraries(nnfusion_engine_pass_graph PRIVATE nnfusion_common
gflags
)

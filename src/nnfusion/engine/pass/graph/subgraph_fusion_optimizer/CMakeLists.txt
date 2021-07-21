# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.


set(SRC
   subgraph_fusion_optimizer.cpp
   conv_elementwise_fusion_optimizer.cpp
  
)

add_library(nnfusion_engine_pass_graph_subgraph_fusion STATIC ${SRC})
target_include_directories(nnfusion_engine_pass_graph_subgraph_fusion SYSTEM PUBLIC
    ${GLOBAL_INCLUDE_PATH}
)
target_compile_options(nnfusion_engine_pass_graph_subgraph_fusion PRIVATE "-fPIC")

target_link_libraries(nnfusion_engine_pass_graph_subgraph_fusion PRIVATE nnfusion_common nnfusion_graph
gflags
)

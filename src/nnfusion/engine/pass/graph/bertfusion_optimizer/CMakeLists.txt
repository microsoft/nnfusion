# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.


set(SRC
   bertfusion_optimizer.cpp
   attention_fusion_optimizer.cpp
   gelu_fusion_optimizer.cpp
   layernorm_fusion_optimizer.cpp
   embedlayernorm_fusion_optimizer.cpp
   skiplayernorm_fusion_optimizer.cpp
   matmuladd_fusion_optimizer.cpp
)

add_library(nnfusion_engine_pass_graph_bertfusion STATIC ${SRC})
target_include_directories(nnfusion_engine_pass_graph_bertfusion SYSTEM PUBLIC
    ${GLOBAL_INCLUDE_PATH}
)
target_compile_options(nnfusion_engine_pass_graph_bertfusion PRIVATE "-fPIC")

target_link_libraries(nnfusion_engine_pass_graph_bertfusion PRIVATE nnfusion_common
gflags
)

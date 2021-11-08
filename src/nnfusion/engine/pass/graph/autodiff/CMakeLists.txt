# Microsoft (c) 2020, NNFusion Team

file(GLOB SRC *.cpp)

add_library(nnfusion_engine_pass_graph_autodiff STATIC ${SRC})
target_include_directories(nnfusion_engine_pass_graph_autodiff SYSTEM PUBLIC
    ${NGRAPH_INCLUDE_PATH}
)
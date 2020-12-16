# Microsoft (c) 2020, NNFusion Team

file(GLOB SRC *.cpp)

add_library(nnfusion_engine_pass_graph_autodiff STATIC ${SRC})
target_include_directories(nnfusion_engine_pass_graph_autodiff SYSTEM PUBLIC
    ${NGRAPH_INCLUDE_PATH}
)
target_compile_options(nnfusion_engine_pass_graph_autodiff PRIVATE "-fPIC")

# Microsoft (c) 2019, NNFusion Team

add_subdirectory(autodiff)

set(SRC
    common_subexpression_elimination_pass.cpp
    autodiff_pass.cpp
)

add_library(ngraph_engine_pass_graph STATIC ${SRC})
target_include_directories(ngraph_engine_pass_graph SYSTEM PUBLIC
    ${GLOBAL_INCLUDE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}
)

target_compile_options(ngraph_engine_pass_graph PRIVATE "-fPIC")

target_link_libraries(ngraph_engine_pass_graph PRIVATE ${WholeArchiveFlag} nnfusion_engine_pass_graph_autodiff ${NoWholeArchiveFlag}
)


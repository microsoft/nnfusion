# Microsoft (c) 2019, NNFusion Team

set(SRC
    # graph.cpp
    # gnode.cpp
    # gedge.cpp
    # graph_util.cpp
    util/autobroadcast.cpp
    util/numpy_transpose.cpp
)

add_library(ngraph_graph STATIC ${SRC})
target_include_directories(ngraph_graph SYSTEM PUBLIC
    ${GLOBAL_INCLUDE_PATH}
)
target_compile_options(ngraph_graph PRIVATE "-fPIC")

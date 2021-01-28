# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.


set(SRC
    graph.cpp
    gnode.cpp
    gedge.cpp
    graph_util.cpp
    util/autobroadcast.cpp
    util/numpy_transpose.cpp
)

add_library(nnfusion_graph STATIC ${SRC})
target_include_directories(nnfusion_graph SYSTEM PUBLIC
    ${GLOBAL_INCLUDE_PATH}
)
target_link_libraries(nnfusion_graph nnfusion_serialize)
target_compile_options(nnfusion_graph PRIVATE "-fPIC")
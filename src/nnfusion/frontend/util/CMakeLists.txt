# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.


add_library(frontend_util STATIC
        parameter.cpp)

target_include_directories(frontend_util SYSTEM PUBLIC
    ${GLOBAL_INCLUDE_PATH}
)
target_compile_options(frontend_util PRIVATE "-fPIC")

target_link_libraries(frontend_util PRIVATE nnfusion_graph)

set_property(TARGET frontend_util PROPERTY POSITION_INDEPENDENT_CODE ON)

set_property(TARGET frontend_util PROPERTY CXX_STANDARD 11)



# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.


add_subdirectory(graph)

file(GLOB codegen_pass codegen/*.cpp)
file(GLOB tensor_pass tensor/*.cpp)
file(GLOB train_pass train/*.cpp)

set(SRC
    extract_graph_signature.cpp
    ${codegen_pass}
    ${tensor_pass}
    ${train_pass}
)

add_library(nnfusion_engine_pass STATIC ${SRC})
target_include_directories(nnfusion_engine_pass SYSTEM PUBLIC
    ${GLOBAL_INCLUDE_PATH}
)
target_compile_options(nnfusion_engine_pass PRIVATE "-fPIC")
target_link_libraries(nnfusion_engine_pass PRIVATE gflags)

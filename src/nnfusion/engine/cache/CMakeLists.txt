# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.


set(SRC
    manager.cpp
)
add_library(nnfusion_cache_manager STATIC ${SRC})
target_include_directories(nnfusion_cache_manager SYSTEM PUBLIC
    ${GLOBAL_INCLUDE_PATH}
)

target_link_libraries(nnfusion_cache_manager
    nnfusion_graph sqlite3
)
target_compile_options(nnfusion_cache_manager PRIVATE "-fPIC")
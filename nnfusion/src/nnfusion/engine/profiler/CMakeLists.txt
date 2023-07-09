# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.


set(ENGINE_PROFILER
    profiler.cpp
    cuda_runtime.cpp
    rocm_runtime.cpp
    cpu_runtime.cpp
    profiling_runtime.cpp
    binary_utils.cpp
)

add_library(nnfusion_engine_profiler STATIC
    ${ENGINE_PROFILER}
)

target_link_libraries(nnfusion_engine_profiler
    nnfusion_graph
    sqlite3
    ${CMAKE_DL_LIBS}
)

target_include_directories(nnfusion_engine_profiler SYSTEM PUBLIC
    ${GLOBAL_INCLUDE_PATH}
)
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.


# Make pass and op object
add_subdirectory(pass)
add_subdirectory(device)
add_subdirectory(profiler)
add_subdirectory(cache)

# This is the source file shared by devices
set(ENGINE
    engine.cpp
    op.cpp
    memory_allocator.cpp
    async_manager.cpp
    util/file_util.cpp
)

add_library(nnfusion_backend SHARED
    ${ENGINE}
)

target_link_libraries(nnfusion_backend PRIVATE
   # nnfusion_engine_pass
    nnfusion_common
    # nnfusion_ir
    nnfusion_util
    # kernels
    # use ${WholeArchiveFlag} because MacOS' ldtool use different flag.
    ${WholeArchiveFlag}
    nnfusion_ir
    nnfusion_engine_pass
    kernels_registration
    nnfusion_engine_device
    kernels_cuda 
    kernels_cpu
    kernels_hlsl
    nnfusion_cache_manager
    nnfusion_engine_profiler
    ${NoWholeArchiveFlag} 
)

target_link_libraries(nnfusion_backend PUBLIC
    gflags dl
)

install(TARGETS nnfusion_backend
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    PUBLIC_HEADER DESTINATION include
)

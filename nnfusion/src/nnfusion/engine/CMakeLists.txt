# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

set(MEMALLOC
    memory_allocator.cpp
)

add_library(nnfusion_engine_mem STATIC 
    ${MEMALLOC}
)


# Make pass and op object
add_subdirectory(pass)
add_subdirectory(device)
add_subdirectory(profiler)
add_subdirectory(cache)

# This is the source file shared by devices
set(ENGINE
    engine.cpp
    op.cpp
    async_manager.cpp
    util/file_util.cpp
)

add_library(nnfusion_engine_base STATIC 
    ${ENGINE}
)

target_link_libraries(nnfusion_engine_base
   nnfusion_engine_device
   nnfusion_engine_pass
   nnfusion_engine_mem
)
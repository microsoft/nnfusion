# Microsoft (c) 2019, Wenxiang Hu

if (NNFUSION_ENABLE)

    # Make pass and op object
    add_subdirectory(pass)
    add_subdirectory(profiler)

    # This is the source file shared by devices
    set(ENGINE
        interpreter.cpp
        op.cpp
        memory_allocator.cpp
        async_manager.cpp
        external/backend_manager.cpp
        external/backend.cpp
    )

    add_library(nnfusion_backend SHARED
        ${ENGINE}
    )

    set_target_properties(nnfusion_backend PROPERTIES VERSION ${NGRAPH_VERSION})
    target_link_libraries(nnfusion_backend PRIVATE
        nnfusion_engine_pass
        nnfusion_common
        nnfusion_ir
        nnfusion_util
        # kernels
        kernels_registration
        # use ${WholeArchiveFlag} because MacOS' ldtool use different flag.
        ${WholeArchiveFlag}
        kernels_cuda 
        kernels_cpu
        nnfusion_engine_profiler
        ${NoWholeArchiveFlag} 
    )

    target_link_libraries(nnfusion_backend PUBLIC
        gflags dl
    )

    set_target_properties(nnfusion_backend PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${NGRAPH_BUILD_DIR})

    install(TARGETS nnfusion_backend
        LIBRARY DESTINATION "${NGRAPH_INSTALL_LIB}"
        ARCHIVE DESTINATION "${NGRAPH_INSTALL_LIB}"
    )

endif()
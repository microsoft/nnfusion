# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

set (SRC
    nnfusion.cpp
)

add_executable(nnfusion ${SRC})
target_include_directories(nnfusion PRIVATE ${TFPROTO_INCLUDE})
target_link_libraries(nnfusion ${CMAKE_DL_LIBS})

if (TENSORFLOW_FRONTEND)
    target_link_libraries(nnfusion tensorflow_import_interface tensorflow_import tensorflow_proto)
endif()

if (ONNX_FRONTEND)
    target_link_libraries(nnfusion onnx_import_interface onnx_import onnx_proto frontend_util)
endif()

if (TORCHSCRIPT_FRONTEND)
    target_link_libraries(nnfusion torchscript_import_interface torchscript_import frontend_util)
endif()

target_link_libraries(nnfusion 
    -Wl,--whole-archive
    # Common feature
    nnfusion_common
    nnfusion_util
    nnfusion_cache_manager
    nnfusion_operators 
    nnfusion_serialize
    nnfusion_graph
    nnfusion_ir
    # Engine related
    nnfusion_engine_base 
    nnfusion_engine_profiler
    nnfusion_engine_device
    nnfusion_engine_mem
    nnfusion_engine_pass
    nnfusion_engine_pass_graph
    nnfusion_engine_pass_graph_blockfusion
    nnfusion_engine_pass_graph_subgraph_fusion
    nnfusion_engine_pass_graph_autodiff
    # Kernels and Operators
    contrib_custom_operators
    kernels_registration
    kernels_cuda 
    kernels_cpu
    kernels_hlsl
    # Depends
    -Wl,--no-whole-archive
    dl
    # static libraries
    protobuf::libprotobuf
    ${GFLAGS}
    ${SQLITE3}
)

# Always copy templates for every build
add_custom_target(dummy_target ALL DEPENDS always_copy)
add_custom_command(
    OUTPUT always_copy
    POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_SOURCE_DIR}/src/tools/nnfusion/templates ${CMAKE_BINARY_DIR}/src/tools/nnfusion/templates
    POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_SOURCE_DIR}/models/pytorch2onnx/ort_run_frozen.py ${CMAKE_BINARY_DIR}/src/tools/nnfusion/templates/onnx/ort_run_frozen.py
    POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_SOURCE_DIR}/models/pytorch2onnx/increase_precision.py ${CMAKE_BINARY_DIR}/src/tools/nnfusion/templates/onnx/increase_precision.py
    POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_SOURCE_DIR}/thirdparty/eigen ${CMAKE_BINARY_DIR}/src/tools/nnfusion/eigen
    POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_SOURCE_DIR}/thirdparty/threadpool ${CMAKE_BINARY_DIR}/src/tools/nnfusion/threadpool
    POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_SOURCE_DIR}/thirdparty/mlas ${CMAKE_BINARY_DIR}/src/tools/nnfusion/mlas
    POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_SOURCE_DIR}/thirdparty/mkl ${CMAKE_BINARY_DIR}/src/tools/nnfusion/mkl
    POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_SOURCE_DIR}/thirdparty/superscaler ${CMAKE_BINARY_DIR}/src/tools/nnfusion/superscaler
    POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_SOURCE_DIR}/thirdparty/cub ${CMAKE_BINARY_DIR}/src/tools/nnfusion/cub
    POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_SOURCE_DIR}/thirdparty/cutlass ${CMAKE_BINARY_DIR}/src/tools/nnfusion/cutlass
    POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_SOURCE_DIR}/src/tools/nnfusion/templates ${CMAKE_BINARY_DIR}/test/templates
    POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_SOURCE_DIR}/src/tools/nnfusion/extensions ${CMAKE_BINARY_DIR}/extensions
)

install(TARGETS nnfusion
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    PUBLIC_HEADER DESTINATION include
)

install(DIRECTORY ${CMAKE_BINARY_DIR}/src/tools/nnfusion/templates DESTINATION bin)
install(DIRECTORY ${CMAKE_BINARY_DIR}/src/tools/nnfusion/eigen DESTINATION bin)
install(DIRECTORY ${CMAKE_BINARY_DIR}/src/tools/nnfusion/threadpool DESTINATION bin)
install(DIRECTORY ${CMAKE_BINARY_DIR}/src/tools/nnfusion/mlas DESTINATION bin)
install(DIRECTORY ${CMAKE_BINARY_DIR}/src/tools/nnfusion/cub DESTINATION bin)
install(DIRECTORY ${CMAKE_BINARY_DIR}/src/tools/nnfusion/superscaler DESTINATION bin)
install(DIRECTORY ${CMAKE_BINARY_DIR}/test/templates DESTINATION bin)
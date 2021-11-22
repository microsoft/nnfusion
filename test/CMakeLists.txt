# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

add_subdirectory(models)

set(SRC main.cpp)

file(GLOB nnfusion_core_files nnfusion/core/*.cpp)
file(GLOB nnfusion_common_files nnfusion/common/*.cpp)
file(GLOB nnfusion_engine_files nnfusion/engine/*.cpp)
file(GLOB nnfusion_inventory_files nnfusion/inventory/*.cpp)
file(GLOB nnfusion_kernels_files nnfusion/kernels/*.cpp)
file(GLOB nnfusion_testutil_files nnfusion/test_util/*.cpp)
if (ONNX_FRONTEND)
    list(APPEND nnfusion_frontend_files nnfusion/frontend/onnx_import.cpp)
endif()
if (TENSORFLOW_FRONTEND)
    list(APPEND nnfusion_frontend_files nnfusion/frontend/tensorflow_import.cpp)
endif()
if (TORCHSCRIPT_FRONTEND)
    list(APPEND nnfusion_frontend_files nnfusion/frontend/torchscript_import.cpp)
endif()
list(APPEND SRC ${nnfusion_core_files}
    ${nnfusion_engine_files}
    ${nnfusion_common_files}
    ${nnfusion_inventory_files}
    ${nnfusion_kernels_files}
    ${nnfusion_testutil_files}
    ${nnfusion_frontend_files}
    )

add_executable(unit-test ${SRC})

target_include_directories(unit-test PRIVATE ".")

if (ONNX_FRONTEND)
    target_include_directories(unit-test PRIVATE ${ONNXPROTO_INCLUDE})
    target_link_libraries(unit-test PRIVATE onnx_import_interface onnx_import onnx_proto)
endif()

if (TENSORFLOW_FRONTEND)
    target_include_directories(unit-test PRIVATE ${TFPROTO_INCLUDE})
    target_link_libraries(unit-test PRIVATE tensorflow_import_interface tensorflow_import tensorflow_proto)
endif()

if (TORCHSCRIPT_FRONTEND)
    target_include_directories(unit-test PRIVATE ${TORCH_INCLUDE_DIRS})
    target_link_libraries(unit-test PRIVATE torchscript_import_interface torchscript_import frontend_util)
endif()

if (ONNX_FRONTEND)
    target_include_directories(unit-test PRIVATE ${ONNXPROTO_INCLUDE})
    target_link_libraries(unit-test PRIVATE onnx_import_interface onnx_import onnx_proto)
endif()

target_link_libraries(unit-test PRIVATE
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
    protobuf::libprotobuf
    dl
    ${GFLAGS}
    ${SQLITE3}
    GTest::GTest
    Threads::Threads
)

add_custom_target(unit-test-check
    COMMAND ${PROJECT_BINARY_DIR}/test/unit-test \${ARGS}
    DEPENDS unit-test
)

add_custom_target(check
    DEPENDS
    style-check
    unit-test-check
)

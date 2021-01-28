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

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DCURDIR=\\\"${CMAKE_CURRENT_SOURCE_DIR}\\\"")

if(NGRAPH_ADDRESS_SANITIZER)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -fsanitize=address -fno-omit-frame-pointer")
endif()

target_link_libraries(unit-test PRIVATE GTest::GTest Threads::Threads gflags)
target_link_libraries(unit-test PRIVATE ${CMAKE_DL_LIBS})
target_link_libraries(unit-test PRIVATE nnfusion_operators)

if ("${CMAKE_CXX_COMPILER_ID}" MATCHES "^(Apple)?Clang$")
    target_compile_options(unit-test PRIVATE -Wno-undef -Wno-reserved-id-macro)
endif()

target_link_libraries(unit-test PRIVATE nnfusion_backend nnfusion_engine_profiler nnfusion_engine_pass_graph)

target_include_directories(unit-test PRIVATE ${Protobuf_INCLUDE_DIR})

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

if (NGRAPH_ONNXIFI_ENABLE)
    target_include_directories(unit-test SYSTEM PUBLIC ${ONNX_INCLUDE_DIR})
    target_link_libraries(unit-test PRIVATE onnxifi-ngraph)
endif()

add_custom_target(unit-test-check
    COMMAND ${PROJECT_BINARY_DIR}/test/unit-test \${ARGS}
    DEPENDS unit-test
)

add_custom_target(check
    DEPENDS
    style-check
    unit-test-check
)

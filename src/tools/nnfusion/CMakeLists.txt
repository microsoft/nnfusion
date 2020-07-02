# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

set (SRC
    nnfusion.cpp
)

add_executable(nnfusion ${SRC})
target_include_directories(nnfusion PRIVATE ${TFPROTO_INCLUDE})
target_link_libraries(nnfusion ${CMAKE_DL_LIBS})
if (NNFUSION_TENSORFLOW_IMPORT_ENABLE)
    target_link_libraries(nnfusion tensorflow_import_interface tensorflow_import tensorflow_proto)
endif()
if (NGRAPH_ONNX_IMPORT_ENABLE)
    target_link_libraries(nnfusion onnx_import)
endif()
if (NNFUSION_ENABLE)
    target_link_libraries(nnfusion nnfusion_backend nnfusion_engine_pass_graph nnfusion_operators)
endif()
target_link_libraries(nnfusion gflags)

install(TARGETS nnfusion RUNTIME DESTINATION ${NGRAPH_INSTALL_BIN})

add_custom_command(
    TARGET nnfusion
    POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_SOURCE_DIR}/src/tools/nnfusion/templates ${CMAKE_BINARY_DIR}/src/tools/nnfusion/templates
    POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_SOURCE_DIR}/thirdparty/mlas ${CMAKE_BINARY_DIR}/src/tools/nnfusion/mlas
    POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_SOURCE_DIR}/src/tools/nnfusion/templates ${CMAKE_BINARY_DIR}/test/templates
)

# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

add_library(tensorflow_import_interface STATIC
        tensorflow.cpp
        tensorflow.hpp
)

add_library(tensorflow_import STATIC
        util/graph_convert.cpp
        util/graph_convert.hpp
	util/util.cpp
        util/util.hpp
        util/bcast.hpp
        util/bcast.cpp
        ops/const.cpp
        ops/const.hpp
        tensorflow_base.cpp
        tensorflow_base.hpp
)

add_dependencies(tensorflow_import_interface tensorflow_proto)
add_dependencies(tensorflow_import tensorflow_import_interface)

if (NOT NGRAPH_USE_SYSTEM_PROTOBUF)
    add_dependencies(tensorflow_import_interface protobuf::libprotobuf)
    add_dependencies(tensorflow_import protobuf::libprotobuf)
endif()

set_property(TARGET tensorflow_import PROPERTY POSITION_INDEPENDENT_CODE ON)
target_include_directories(tensorflow_import PRIVATE ${TFPROTO_INCLUDE} ${CMAKE_CURRENT_SOURCE_DIR} ${GLOBAL_INCLUDE_PATH}
    SYSTEM PRIVATE ${Protobuf_INCLUDE_DIR})
target_link_libraries(tensorflow_import PRIVATE ${Protobuf_LIBRARIES} ${ONNX_PROTO_LIBRARY} nnfusion_operators nnfusion_graph)

set_property(TARGET tensorflow_import_interface PROPERTY POSITION_INDEPENDENT_CODE ON)
target_include_directories(tensorflow_import_interface PRIVATE ${TFPROTO_INCLUDE} {CMAKE_CURRENT_BINARY_DIR} ${GLOBAL_INCLUDE_PATH}
    SYSTEM PRIVATE ${Protobuf_INCLUDE_DIR})

if ("${CMAKE_CXX_COMPILER_ID}" MATCHES "^(Apple)?Clang$")
    target_compile_options(tensorflow_import PRIVATE -Wno-undef -Wno-reserved-id-macro -Wno-switch-enum
            -Wno-extended-offsetof -Wno-shorten-64-to-32 -Wno-unused-macros -Wno-missing-variable-declarations
            -Wno-unused-private-field -Wno-shadow -Wno-deprecated)
    target_compile_options(tensorflow_import_interface PRIVATE -Wno-undef -Wno-reserved-id-macro -Wno-switch-enum
            -Wno-extended-offsetof -Wno-shorten-64-to-32 -Wno-unused-macros -Wno-missing-variable-declarations
            -Wno-unused-private-field -Wno-shadow -Wno-deprecated)
endif()

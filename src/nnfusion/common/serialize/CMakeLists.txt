# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

FILE(TO_NATIVE_PATH ${CMAKE_CURRENT_SOURCE_DIR} PROTOSRC_PATH)
FOREACH(item nnf_pbtypes nnf_attr_value nnf_tensor_shape nnf_node_def nnf_graph_def)
  EXECUTE_PROCESS(COMMAND ${PROTOBUF_PROTOC_EXECUTABLE} --proto_path=${PROTOSRC_PATH} --cpp_out=${CMAKE_CURRENT_BINARY_DIR} ${item}.proto)
  FILE(TO_NATIVE_PATH ${item}.pb.h proto_header)
  FILE(TO_NATIVE_PATH ${item}.pb.cc proto_source)
  list(APPEND SRC ${proto_header} ${proto_source})
ENDFOREACH(item)

include_directories(${CMAKE_CURRENT_BINARY_DIR})
add_library(nnfusion_serialize STATIC ${SRC})
target_include_directories(nnfusion_serialize SYSTEM PUBLIC
    ${GLOBAL_INCLUDE_PATH}
)

target_link_libraries(nnfusion_serialize ${Protobuf_LIBRARIES})

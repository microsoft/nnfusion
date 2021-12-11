# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

FILE(TO_NATIVE_PATH ${PROJECT_SOURCE_DIR}/src/nnfusion/common/serialize PROTOSRC_PATH)
FOREACH(item pbtypes attr_value tensor_shape node_def graph_def)
  EXECUTE_PROCESS(COMMAND ${PROTOBUF_PROTOC_EXECUTABLE} --proto_path=${PROTOSRC_PATH} --python_out=${CMAKE_CURRENT_BINARY_DIR} ${item}.proto)
ENDFOREACH(item)
EXECUTE_PROCESS(COMMAND cp ${CMAKE_CURRENT_SOURCE_DIR}/nnfusion_serialize_tool.py ${CMAKE_CURRENT_BINARY_DIR})
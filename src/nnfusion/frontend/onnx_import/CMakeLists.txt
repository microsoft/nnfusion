# ******************************************************************************
# Copyright (c) 2017-2018 Intel Copyright
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by application law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ******************************************************************************

set(ONNX_OPSET_VERSION 9 CACHE INTERNAL "Supported version of ONNX operator set")

add_library(onnx_import_interface STATIC
        onnx.cpp
        onnx.hpp
)

# file(GLOB ops_source op/*.cpp)

add_library(onnx_import STATIC
        onnx_base.hpp
        onnx_base.cpp
        ops_bridge.hpp
        ops_bridge.cpp
        core/attribute.hpp
        core/attribute.cpp
        core/node.hpp
        core/node.cpp
        op/no.hpp
        op/no.cpp
        op/slice.hpp
        op/slice.cpp
        op/split.hpp
        op/split.cpp
        op/expand.hpp
        op/expand.cpp
        op/gather.hpp
        op/gather.cpp
        op/softmax.hpp
        op/softmax.cpp
        op/squeeze.hpp
        op/squeeze.cpp
        op/unsqueeze.hpp
        op/unsqueeze.cpp
        op/matmul.hpp
        op/matmul.cpp
        op/reshape.hpp
        op/reshape.cpp
        op/transpose.hpp
        op/transpose.cpp
        op/gemm.hpp
        op/gemm.cpp
        op/resize.hpp
        op/resize.cpp
        op/conv.hpp
        op/conv.cpp
        op/shape.hpp
        op/shape.cpp
        op/conv_trans.hpp
        op/conv_trans.cpp
        op/hard_sigmoid.hpp
        op/hard_sigmoid.cpp
        op/where.hpp
        op/where.cpp
        op/embed_layer_norm.hpp
        op/embed_layer_norm.cpp
        op/layer_norm.hpp
        op/layer_norm.cpp
        op/instance_norm.hpp
        op/instance_norm.cpp
        op/attention.hpp
        op/attention.cpp
        op/scatternd.hpp
        op/scatternd.cpp
        op/depth_to_space.hpp
        op/depth_to_space.cpp
        op/roll.hpp
        op/roll.cpp
        op/lstm.hpp
        op/lstm.cpp
        op/gru.hpp
        op/gru.cpp
        op/einsum.hpp
        op/einsum.cpp
        # ${ops_source}
        util/broadcasting.cpp
        util/graph_convert.cpp
        util/graph_convert.hpp
        util/reshape.hpp
        util/reshape.cpp
        util/util.hpp
        util/util.cpp)

add_dependencies(onnx_import_interface onnx_proto)
add_dependencies(onnx_import onnx_import_interface)

if (NOT NGRAPH_USE_SYSTEM_PROTOBUF)
    #add_dependencies(onnx_import_interface protobuf::libprotobuf)
    #add_dependencies(onnx_import protobuf::libprotobuf)
endif()

set_property(TARGET onnx_import PROPERTY POSITION_INDEPENDENT_CODE ON)
target_include_directories(onnx_import PRIVATE ${ONNXPROTO_INCLUDE} ${CMAKE_CURRENT_SOURCE_DIR} ${GLOBAL_INCLUDE_PATH}
    SYSTEM PRIVATE ${Protobuf_INCLUDE_DIR})
target_compile_definitions(onnx_import PRIVATE ONNX_OPSET_VERSION=${ONNX_OPSET_VERSION})

set_property(TARGET onnx_import_interface PROPERTY POSITION_INDEPENDENT_CODE ON)
target_include_directories(onnx_import_interface PRIVATE ${ONNXPROTO_INCLUDE} ${CMAKE_CURRENT_BINARY_DIR} ${GLOBAL_INCLUDE_PATH}
    SYSTEM PRIVATE  ${Protobuf_INCLUDE_DIR})

if ("${CMAKE_CXX_COMPILER_ID}" MATCHES "^(Apple)?Clang$")
    target_compile_options(onnx_import_interface PRIVATE -Wno-undef -Wno-reserved-id-macro -Wno-switch-enum
            -Wno-extended-offsetof -Wno-shorten-64-to-32 -Wno-unused-macros -Wno-missing-variable-declarations
            -Wno-unused-private-field -Wno-shadow -Wno-deprecated)
    target_compile_options(onnx_import PRIVATE -Wno-undef -Wno-reserved-id-macro -Wno-switch-enum
            -Wno-extended-offsetof -Wno-shorten-64-to-32 -Wno-unused-macros -Wno-missing-variable-declarations
            -Wno-unused-private-field -Wno-shadow -Wno-deprecated)
endif()

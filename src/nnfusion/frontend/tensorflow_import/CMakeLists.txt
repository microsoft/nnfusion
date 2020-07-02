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
    # add_dependencies(onnx_import_interface onnx::libonnx_proto)
endif()

set_property(TARGET tensorflow_import PROPERTY POSITION_INDEPENDENT_CODE ON)
target_include_directories(tensorflow_import PRIVATE ${TFPROTO_INCLUDE} ${CMAKE_CURRENT_SOURCE_DIR} ${NGRAPH_INCLUDE_PATH}
    SYSTEM PRIVATE ${Protobuf_INCLUDE_DIR})
target_link_libraries(tensorflow_import PRIVATE ${Protobuf_LIBRARIES} ${ONNX_PROTO_LIBRARY} nnfusion_operators nnfusion_graph)

# target_compile_definitions(onnx_import PRIVATE ONNX_OPSET_VERSION=${ONNX_OPSET_VERSION})

set_property(TARGET tensorflow_import_interface PROPERTY POSITION_INDEPENDENT_CODE ON)
target_include_directories(tensorflow_import_interface PRIVATE ${TFPROTO_INCLUDE} {CMAKE_CURRENT_BINARY_DIR} ${NGRAPH_INCLUDE_PATH}
    SYSTEM PRIVATE ${Protobuf_INCLUDE_DIR})

if ("${CMAKE_CXX_COMPILER_ID}" MATCHES "^(Apple)?Clang$")
    target_compile_options(tensorflow_import PRIVATE -Wno-undef -Wno-reserved-id-macro -Wno-switch-enum
            -Wno-extended-offsetof -Wno-shorten-64-to-32 -Wno-unused-macros -Wno-missing-variable-declarations
            -Wno-unused-private-field -Wno-shadow -Wno-deprecated)
    target_compile_options(tensorflow_import_interface PRIVATE -Wno-undef -Wno-reserved-id-macro -Wno-switch-enum
            -Wno-extended-offsetof -Wno-shorten-64-to-32 -Wno-unused-macros -Wno-missing-variable-declarations
            -Wno-unused-private-field -Wno-shadow -Wno-deprecated)
endif()

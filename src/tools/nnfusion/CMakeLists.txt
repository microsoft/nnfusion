# ******************************************************************************
# Copyright 2017-2018 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ******************************************************************************

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

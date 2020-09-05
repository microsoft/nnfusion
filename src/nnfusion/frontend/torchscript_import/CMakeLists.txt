# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.


add_library(torchscript_import_interface STATIC
        torchscript.cpp
        torchscript.hpp
)

add_library(torchscript_import STATIC
        util/graph_convert.cpp
        util/graph_convert.hpp
        util/util.cpp
        util/util.hpp
        ops/const.cpp
        ops/const.hpp
        torchscript_base.cpp
        torchscript_base.hpp
)

add_dependencies(torchscript_import_interface protobuf::libprotobuf)
add_dependencies(torchscript_import torchscript_import_interface)

target_link_libraries(torchscript_import_interface PRIVATE nnfusion_graph libtorch)
target_link_libraries(torchscript_import PRIVATE nnfusion_graph nnfusion_operators libtorch)

set_property(TARGET torchscript_import_interface PROPERTY POSITION_INDEPENDENT_CODE ON)
set_property(TARGET torchscript_import_interface PROPERTY CXX_STANDARD 11)



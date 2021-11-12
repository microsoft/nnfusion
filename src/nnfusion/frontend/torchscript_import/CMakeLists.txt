# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

list(APPEND CMAKE_PREFIX_PATH ${TORCH_PREFIX})
find_package(Torch REQUIRED)

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
        torchscript_base.hpp
)

target_include_directories(torchscript_import SYSTEM PUBLIC ${TORCH_INCLUDE_DIRS})
# Currently libtorch uses CXX14 templates, might be upgraded in the future
set_property(TARGET torchscript_import PROPERTY CXX_STANDARD 14)
set_property(TARGET torchscript_import_interface PROPERTY CXX_STANDARD 14)

set_property(TARGET torchscript_import_interface PROPERTY POSITION_INDEPENDENT_CODE ON)



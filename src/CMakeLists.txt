# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DIN_NGRAPH_LIBRARY")

add_subdirectory(resource)
add_subdirectory(nnfusion)

if (NGRAPH_TOOLS_ENABLE)
    add_subdirectory(tools)
    message(STATUS "tools enabled")
else()
    message(STATUS "tools disabled")
endif()

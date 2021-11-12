# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.


add_library(frontend_util STATIC
        parameter.cpp)

target_include_directories(frontend_util SYSTEM PUBLIC
    ${GLOBAL_INCLUDE_PATH}
)
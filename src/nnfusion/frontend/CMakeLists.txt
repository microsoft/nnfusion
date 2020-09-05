# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.


if (TENSORFLOW_FRONTEND)
    add_subdirectory(tensorflow_import)
endif()

if (TORCHSCRIPT_FRONTEND)
    add_subdirectory(torchscript_import)
endif()

add_subdirectory(util)
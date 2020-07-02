# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

if (NNFUSION_ENABLE)
    add_subdirectory(common)
    add_subdirectory(engine)
    add_subdirectory(core)
    add_subdirectory(util)
    add_subdirectory(frontend)

endif()
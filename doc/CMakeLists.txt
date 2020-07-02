# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

if ("${NGRAPH_BUILD_DOCS}" MATCHES "^ON$")
    add_custom_target( docs
        COMMENT "Build all of the documentation types selected during CMake configuration."
        )

    add_subdirectory( doxygen )
else()
    add_custom_target( docs
        COMMAND echo
        COMMAND echo "The 'docs' target is disabled. To enable the building of documentation, re-run cmake with the option -DNGRAPH_BUILD_DOCS=ON."
        COMMAND echo
        COMMAND false
        VERBATIM
        )
endif()

# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

find_package(Doxygen REQUIRED)

if ("${NGRAPH_DOXYGEN_WARN_IF_UNDOCUMENTED}" MATCHES "^ON$")
    set(DOXYGEN_WARN_IF_UNDOCUMENTED YES)
else()
    set(DOXYGEN_WARN_IF_UNDOCUMENTED NO)
endif()

if ("${NGRAPH_DOXYGEN_QUIET}" MATCHES "^ON$")
    set(DOXYGEN_QUIET YES)
else()
    set(DOXYGEN_QUIET NO)
endif()

set(DOXYGEN_IN "${CMAKE_CURRENT_SOURCE_DIR}/Doxyfile.in")
set(DOXYGEN_OUT "${CMAKE_CURRENT_BINARY_DIR}/Doxyfile")
configure_file("${DOXYGEN_IN}" "${DOXYGEN_OUT}" @ONLY)

add_custom_target(doxygen-docs
    ALL
    COMMAND "${DOXYGEN_EXECUTABLE}" "${DOXYGEN_OUT}"
    WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
    COMMENT "Generating documentation with Doxygen"
    VERBATIM )

add_dependencies( docs doxygen-docs )

install(
    DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/html/"
    DESTINATION "${NGRAPH_INSTALL_DOC}/api-reference/html"
    OPTIONAL
    )

install(
    DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/latex/"
    DESTINATION "${NGRAPH_INSTALL_DOC}/api-reference/latex"
    OPTIONAL
    )

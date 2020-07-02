# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

# The following variables are optionally searched for defaults
#  CLDNN_ROOT_DIR:     Base directory where all Intel clDNN components are found
#
# The following are set after configuration is done:
#  CLDNN_FOUND
#  CLDNN_INCLUDE_DIRS
#  CLDNN_LIBRARIES
#  CLDNN_LIBRARY_DIRS

include(FindPackageHandleStandardArgs)

set(CLDNN_ROOT_DIR "" CACHE PATH "Folder contains Intel clDNN")

find_path(CLDNN_INCLUDE_DIR CPP/cldnn_defs.h
    HINTS ${CLDNN_ROOT_DIR}
    PATH_SUFFIXES api include/clDNN)

find_library(CLDNN_LIBRARY clDNN64
    HINTS ${CLDNN_ROOT_DIR}
    PATH_SUFFIXES build/out/Linux64/Release build/out/Linux64/Debug lib)

find_package_handle_standard_args(
    clDNN DEFAULT_MSG CLDNN_INCLUDE_DIR CLDNN_LIBRARY)

if(CLDNN_FOUND)
    set(CLDNN_INCLUDE_DIRS ${CLDNN_INCLUDE_DIR})
    set(CLDNN_LIBRARIES ${CLDNN_LIBRARY})
    message(STATUS "Found Intel clDNN: (include: ${CLDNN_INCLUDE_DIR}, library: ${CLDNN_LIBRARY})")
    mark_as_advanced(CLDNN_ROOT_DIR CLDNN_LIBRARY CLDNN_INCLUDE_DIR)
endif()

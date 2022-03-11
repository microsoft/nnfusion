# Find the CUDNN libraries
#
# The following variables are optionally searched for defaults
#  CUDNN_ROOT: Base directory where CUDNN is found
#  CUDNN_INCLUDE_DIR: Directory where CUDNN header is searched for
#  CUDNN_LIBRARY: Directory where CUDNN library is searched for
#  CUDNN_STATIC: Are we looking for a static library? (default: no)
#
# The following are set after configuration is done:
#  CUDNN_FOUND
#  CUDNN_INCLUDE_PATH
#  CUDNN_LIBRARY_PATH
#

set(CUDNN_ROOT $ENV{CUDNN_ROOT_DIR} CACHE PATH "Folder containing NVIDIA cuDNN")
if (DEFINED $ENV{CUDNN_ROOT_DIR})
    message(WARNING "CUDNN_ROOT_DIR is deprecated. Please set CUDNN_ROOT instead.")
endif()
list(APPEND CUDNN_ROOT $ENV{CUDNN_ROOT_DIR} ${CUDA_TOOLKIT_ROOT_DIR})

# Compatible layer for CMake <3.12. CUDNN_ROOT will be accounted in for searching paths and libraries for CMake >=3.12.
list(APPEND CMAKE_PREFIX_PATH ${CUDNN_ROOT})

set(CUDNN_INCLUDE_DIR $ENV{CUDNN_INCLUDE_DIR} CACHE PATH "Folder containing NVIDIA cuDNN header files")

find_path(CUDNN_INCLUDE_PATH cudnn.h
    HINTS ${CUDNN_INCLUDE_DIR}
    PATH_SUFFIXES cuda/include cuda include
    REQUIRED)

option(CUDNN_STATIC "Look for static CUDNN" OFF)
if (CUDNN_STATIC)
    set(CUDNN_LIBNAME "libcudnn_static.a")
else()
    set(CUDNN_LIBNAME "cudnn")
endif()

set(CUDNN_LIBRARY $ENV{CUDNN_LIBRARY} CACHE PATH "Path to the cudnn library file (e.g., libcudnn.so)")
if (CUDNN_LIBRARY MATCHES ".*cudnn_static.a" AND NOT CUDNN_STATIC)
    message(WARNING "CUDNN_LIBRARY points to a static library (${CUDNN_LIBRARY}) but CUDNN_STATIC is OFF.")
endif()

find_library(CUDNN_LIBRARY_PATH ${CUDNN_LIBNAME}
    PATHS ${CUDNN_LIBRARY}
    PATH_SUFFIXES lib lib64 cuda/lib cuda/lib64 lib/x64
    REQUIRED)

set(file "${PROJECT_BINARY_DIR}/detect_cudnn_version.cc")
file(WRITE ${file} "
    #include <iostream>
    #include \"${CUDNN_INCLUDE_PATH}/cudnn.h\"
    int main()
    {
        std::cout << CUDNN_MAJOR << '.' << CUDNN_MINOR << '.' << CUDNN_PATCHLEVEL;
        int x = cudnnGetVersion();
        return x == CUDNN_VERSION;
    }
")
try_run(CUDNN_VERSION_MATCHED compile_result ${PROJECT_BINARY_DIR} ${file}
    RUN_OUTPUT_VARIABLE CUDNN_VERSION
    CMAKE_FLAGS  "-DINCLUDE_DIRECTORIES=${CUDAToolkit_INCLUDE_DIR}"
    LINK_LIBRARIES ${CUDNN_LIBRARY_PATH})
if (NOT CUDNN_VERSION_MATCHED)
    message(FATAL_ERROR "Found CUDNN header version and library version do not match! \
        (include: ${CUDNN_INCLUDE_PATH}, library: ${CUDNN_LIBRARY_PATH}). Please set CUDNN_ROOT manually.")
endif()
message(STATUS "CUDNN version: ${CUDNN_VERSION}")

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(
    CUDNN
    REQUIRED_VARS CUDNN_LIBRARY_PATH CUDNN_INCLUDE_PATH
    VERSION_VAR CUDNN_VERSION)

mark_as_advanced(CUDNN_ROOT CUDNN_INCLUDE_DIR CUDNN_LIBRARY)

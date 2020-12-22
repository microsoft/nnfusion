execute_process(
  COMMAND
    python3 -c
    "import superscaler as _; print(_.__path__[0])"
  OUTPUT_VARIABLE SUPERSCALER_INSTALLATION_PATH
  OUTPUT_STRIP_TRAILING_WHITESPACE)
message(STATUS ${SUPERSCALER_INSTALLATION_PATH})

find_path(
  SUPERSCALER_INCLUDE_DIR
  NAMES superscaler.h
  HINTS ${CMAKE_CURRENT_SOURCE_DIR}/superscaler)

find_library(
  SUPERSCALER_LIBRARY
  NAMES superscaler_pywrap
  HINTS ${SUPERSCALER_INSTALLATION_PATH}
  PATH_SUFFIXES lib)

include(FindPackageHandleStandardArgs)
# handle the QUIETLY and REQUIRED arguments and set VILLASNODE_FOUND to TRUE if
# all listed variables are TRUE
find_package_handle_standard_args(superscaler DEFAULT_MSG SUPERSCALER_LIBRARY)

mark_as_advanced(SUPERSCALER_INCLUDE_DIR SUPERSCALER_LIBRARY)

set(SUPERSCALER_LIBRARIES ${SUPERSCALER_LIBRARY})
set(SUPERSCALER_INCLUDE_DIRS ${SUPERSCALER_INCLUDE_DIR})

add_library(superscaler INTERFACE)
target_include_directories(superscaler SYSTEM
                           INTERFACE ${SUPERSCALER_INCLUDE_DIRS})
target_link_libraries(superscaler INTERFACE ${SUPERSCALER_LIBRARIES})

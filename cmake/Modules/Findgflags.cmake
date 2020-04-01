  
add_library(gflags UNKNOWN IMPORTED)

find_path(gflags_INCLUDE_DIR gflags/gflags.h)
mark_as_advanced(gflags_INCLUDE_DIR)

find_library(gflags_LIBRARY gflags)
mark_as_advanced(gflags_LIBRARY)

find_package_handle_standard_args(gflags DEFAULT_MSG
    gflags_INCLUDE_DIR
    gflags_LIBRARY
    )

set_target_properties(gflags PROPERTIES
    INTERFACE_INCLUDE_DIRECTORIES ${gflags_INCLUDE_DIR}
    IMPORTED_LOCATION ${gflags_LIBRARY}
    )

if (NOT MSVC)
  set_target_properties(gflags PROPERTIES
      INTERFACE_LINK_LIBRARIES "pthread"
      )
endif ()
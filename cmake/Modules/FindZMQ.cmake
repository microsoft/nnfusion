# - Try to find ZMQ
# Once done this will define
# ZMQ_FOUND - System has ZMQ
# ZMQ_INCLUDE_DIRS - The ZMQ include directories
# ZMQ_LIBRARIES - The libraries needed to use ZMQ
# ZMQ_DEFINITIONS - Compiler switches required for using ZMQ

find_path ( ZMQ_INCLUDE_DIR zmq.h HINTS ${ZMQ_ROOT}/include )
find_library ( ZMQ_LIBRARY NAMES zmq HINTS ${ZMQ_BUILD}/lib )

set ( ZMQ_LIBRARIES ${ZMQ_LIBRARY} )
set ( ZMQ_INCLUDE_DIRS ${ZMQ_INCLUDE_DIR} )

if (DEFINED ZMQ_LIBRARIES AND DEFINED ZMQ_INCLUDE_DIRS)
    set(file "${PROJECT_BINARY_DIR}/detect_zeromq_version.cc")
    file(WRITE ${file} "
        #include <iostream>
        #include \"${ZMQ_INCLUDE_DIRS}/zmq.h\"
        int main()
        {
            std::cout << ZMQ_VERSION_MAJOR << '.' << ZMQ_VERSION_MINOR << '.' << ZMQ_VERSION_PATCH;
            int x, y, z;
            zmq_version(&x, &y, &z);
            return x == ZMQ_VERSION_MAJOR && y == ZMQ_VERSION_MINOR && z == ZMQ_VERSION_PATCH;
        }
    ")
    try_run(ZMQ_VERSION_MATCHED compile_result ${PROJECT_BINARY_DIR} ${file}
        RUN_OUTPUT_VARIABLE ZMQ_VERSION
        LINK_LIBRARIES ${ZMQ_LIBRARIES})
    if (NOT ZMQ_VERSION_MATCHED)
        message(WARNING "Found ZMQ header version and library version do not match! \
            (include: ${ZMQ_INCLUDE_DIRS}, library: ${ZMQ_LIBRARIES}). Please set ZMQ_ROOT and ZMQ_BUILD carefully.")
        unset(ZMQ_INCLUDE_DIRS)
        unset(ZMQ_LIBRARIES)
        unset(ZMQ_VERSION)
    else ()
        message(STATUS "ZMQ version: ${ZMQ_VERSION}")
    endif()
endif()

include ( FindPackageHandleStandardArgs )
# handle the QUIETLY and REQUIRED arguments and set ZMQ_FOUND to TRUE
# if all listed variables are TRUE
find_package_handle_standard_args (
    ZMQ
    REQUIRED_VARS ZMQ_LIBRARIES ZMQ_INCLUDE_DIRS
    VERSION_VAR ZMQ_VERSION)

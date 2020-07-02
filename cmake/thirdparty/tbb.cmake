# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

#------------------------------------------------------------------------------
# Fetch and configure TBB
#------------------------------------------------------------------------------

if(NGRAPH_TBB_ENABLE)

    include(ExternalProject)
    ExternalProject_Add(
        ext_tbb
        SOURCE_DIR "${NNFUSION_THIRDPARTY_FOLDER}/tbb"
        CONFIGURE_COMMAND ""
        BUILD_COMMAND ""
        UPDATE_COMMAND ""
        INSTALL_COMMAND ""
    )

    execute_process(COMMAND "${CMAKE_COMMAND}" -G "${CMAKE_GENERATOR}" ${NNFUSION_THIRDPARTY_FOLDER}/tbb
        WORKING_DIRECTORY "${NNFUSION_THIRDPARTY_FOLDER}/build/tbb")
    execute_process(COMMAND "${CMAKE_COMMAND}" --build ${NNFUSION_THIRDPARTY_FOLDER}/tbb
        WORKING_DIRECTORY "${NNFUSION_THIRDPARTY_FOLDER}/build/tbb")
    set(TBB_ROOT ${NNFUSION_THIRDPARTY_FOLDER}/tbb)
endif()

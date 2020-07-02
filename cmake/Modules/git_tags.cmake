# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

function(NGRAPH_GET_CURRENT_HASH)
    find_package(Git REQUIRED)
    execute_process(
        COMMAND ${GIT_EXECUTABLE} rev-parse --verify HEAD
        RESULT_VARIABLE result
        OUTPUT_VARIABLE HASH
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
        ERROR_QUIET)

    string(STRIP ${HASH} HASH)
    set(NGRAPH_CURRENT_HASH ${HASH} PARENT_SCOPE)
endfunction()

function(NGRAPH_GET_TAG_OF_CURRENT_HASH)
    find_package(Git REQUIRED)
    execute_process(
        COMMAND ${GIT_EXECUTABLE} ls-remote --tags
        RESULT_VARIABLE RESULT
        OUTPUT_VARIABLE TAG_LIST
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
        ERROR_QUIET)

    NGRAPH_GET_CURRENT_HASH()

    if (NOT ${TAG_LIST} STREQUAL "")
        string(REGEX MATCH "${NGRAPH_CURRENT_HASH}\t[^\r\n]*" TAG ${TAG_LIST})
        set(FINAL_TAG ${TAG})
        if (NOT "${TAG}" STREQUAL "")
            string(REGEX REPLACE "${NGRAPH_CURRENT_HASH}\trefs/tags/(.*)" "\\1" FINAL_TAG ${TAG})
        endif()
    else()
        set(FINAL_TAG "")
    endif()
    set(NGRAPH_CURRENT_RELEASE_TAG ${FINAL_TAG} PARENT_SCOPE)
endfunction()

function(NGRAPH_GET_MOST_RECENT_TAG)
    find_package(Git REQUIRED)
    execute_process(
        COMMAND ${GIT_EXECUTABLE} describe --tags --abbrev=0
        RESULT_VARIABLE RESULT
        OUTPUT_VARIABLE TAG
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
        ERROR_QUIET)

    if (NOT ${TAG} STREQUAL "")
        string(STRIP ${TAG} TAG)
    endif()
    set(NGRAPH_MOST_RECENT_RELEASE_TAG ${TAG} PARENT_SCOPE)
endfunction()

function(NGRAPH_GET_VERSION_LABEL)
    NGRAPH_GET_TAG_OF_CURRENT_HASH()
    set(NGRAPH_VERSION_LABEL ${NGRAPH_CURRENT_RELEASE_TAG} PARENT_SCOPE)
    if ("${NGRAPH_CURRENT_RELEASE_TAG}" STREQUAL "")
        NGRAPH_GET_CURRENT_HASH()
        NGRAPH_GET_MOST_RECENT_TAG()
        string(SUBSTRING "${NGRAPH_CURRENT_HASH}" 0 7 HASH)
        if (NOT ${NGRAPH_MOST_RECENT_RELEASE_TAG} STREQUAL "")
            set(NGRAPH_VERSION_LABEL "${NGRAPH_MOST_RECENT_RELEASE_TAG}+${HASH}" PARENT_SCOPE)
        else()
            set(NGRAPH_VERSION_LABEL "?.?.?+${HASH}" PARENT_SCOPE)
        endif()
    endif()
endfunction()

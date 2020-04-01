# ******************************************************************************
# Copyright 2017-2018 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ******************************************************************************

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

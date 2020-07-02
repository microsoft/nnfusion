# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

# function check existence of cmake and environment variable and read their value
# If the variable is not defined then the supplied default value used
# example:
#    ngraph_var(NGRAPH_VARIABLE_NAME DEFAULT "OFF")
#        checks if variable NGRAPH_VARIABLE_NAME was passed in via the cmake command line
#            if found then passed in value is used
#            else checks for the existence of the environment variable NGRAPH_VARIABLE_NAME
#                if found it's value is used
#        if none of the above then the default value is used
function(NGRAPH_VAR)
    set(options)
    set(oneValueArgs DEFAULT)
    set(multiValueArgs)
    cmake_parse_arguments(NGRAPH_VAR "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})
    if (NOT DEFINED ${NGRAPH_VAR_UNPARSED_ARGUMENTS})
        set(${NGRAPH_VAR_UNPARSED_ARGUMENTS} ${NGRAPH_VAR_DEFAULT} PARENT_SCOPE)
        if(DEFINED ENV{${NGRAPH_VAR_UNPARSED_ARGUMENTS}})
            set(${NGRAPH_VAR_UNPARSED_ARGUMENTS} $ENV{${NGRAPH_VAR_UNPARSED_ARGUMENTS}} PARENT_SCOPE)
        endif()
    endif()
endfunction()

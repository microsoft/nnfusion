#!/bin/bash

# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

#===================================================================================================
# A library of general-purpose Bash functions
#===================================================================================================

declare _intelnervana_bash_lib_SCRIPT_NAME="${BASH_SOURCE[${#BASH_SOURCE[@]} - 1]}"
declare _maint_SCRIPT_DIR="$( cd $(dirname "${_intelnervana_bash_lib_SCRIPT_NAME}") && pwd )"
declare _intelnervana_bash_lib_IS_LOADED=1

bash_lib_get_my_BASH_LINENO() {
    echo "${BASH_LINENO[${#BASH_LINENO[@]} -1 ]}"
}

bash_lib_get_callers_BASH_LINENO() {
    echo "${BASH_LINENO[${#BASH_LINENO[@]} - 2]}"
}

bash_lib_get_my_BASH_SOURCE() {
    echo "${BASH_SOURCE[${#BASH_SOURCE[@]} ]}"
}

bash_lib_get_callers_BASH_SOURCE() {
    echo "${BASH_SOURCE[${#BASH_SOURCE[@]} - 1]}"
}

bash_lib_status() {
    local CONTEXT_STRING="$(basename $(bash_lib_get_callers_BASH_SOURCE))"
    local TEXT_LINE
    local IS_FIRST_LINE=1

    for TEXT_LINE in "${@}"; do
        if (( IS_FIRST_LINE == 1 )); then
            IS_FIRST_LINE=0
            printf "%s STATUS: " "${CONTEXT_STRING}" >&2
        else
            printf "    " >&2
        fi

        printf "%s\n" "${TEXT_LINE}" >&2
    done
}

bash_lib_print_error() {
    local CONTEXT_STRING="$(basename $(bash_lib_get_callers_BASH_SOURCE)):$(bash_lib_get_callers_BASH_LINENO)"
    local TEXT_LINE
    local IS_FIRST_LINE=1

    for TEXT_LINE in "${@}"; do
        if (( IS_FIRST_LINE == 1 )); then
            IS_FIRST_LINE=0
            printf "%s ERROR: " "${CONTEXT_STRING}" >&2
        else
            printf "    " >&2
        fi

        printf "%s\n" "${TEXT_LINE}" >&2
    done
}

bash_lib_die() {
    bash_lib_print_error $@
    exit 1
}

bash_lib_am_sudo_or_root() {
    [ "$EUID" -eq 0 ]
}

if bash_lib_am_sudo_or_root; then
    bash_lib_MAYBE_SUDO=''
else
    bash_lib_MAYBE_SUDO='sudo --set-home'
fi


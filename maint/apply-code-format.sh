
#!/bin/bash
set -e
set -u

# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

# NOTE: The results of `clang-format` depend _both_ of the following factors:
# - The `.clang-format` file, and
# - The particular version of the `clang-format` program being used.
#
# For this reason, this script specifies the exact version of clang-format to be used.

declare CLANG_FORMAT_BASENAME="clang-format-3.9"
declare REQUIRED_CLANG_FORMAT_VERSION=3.9

declare THIS_SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

source "${THIS_SCRIPT_DIR}/bash_lib.sh"
source "${THIS_SCRIPT_DIR}/clang_format_lib.sh"

declare CLANG_FORMAT_PROG
if ! CLANG_FORMAT_PROG="$(which "${CLANG_FORMAT_BASENAME}")"; then
    bash_lib_die "Unable to find program ${CLANG_FORMAT_BASENAME}" >&2
fi

clang_format_lib_verify_version "${CLANG_FORMAT_PROG}" "${REQUIRED_CLANG_FORMAT_VERSION}"
bash_lib_status "Verified that '${CLANG_FORMAT_PROG}' has version '${REQUIRED_CLANG_FORMAT_VERSION}'"

pushd "${THIS_SCRIPT_DIR}/.."

declare PYBIND_WRAPPER="python/pyngraph"

declare ROOT_SUBDIR
for ROOT_SUBDIR in src doc/examples test ${PYBIND_WRAPPER}; do
    if ! [[ -d "${ROOT_SUBDIR}" ]]; then
	    bash_lib_status "In directory '$(pwd)', no subdirectory named '${ROOT_SUBDIR}' was found."
    else
        bash_lib_status "About to format C/C++ code in directory tree '$(pwd)/${ROOT_SUBDIR}' ..."

        # Note that we restrict to "-type f" to exclude symlinks. Emacs sometimes
        # creates dangling symlinks with .cpp/.hpp suffixes as a sort of locking
        # mechanism, and this confuses clang-format.
        find "${ROOT_SUBDIR}" -type f -and \( -name '*.cpp' -or -name '*.hpp' \) | xargs "${CLANG_FORMAT_PROG}" -i -style=file
        find "${ROOT_SUBDIR}" -type f -and \( -name '*.cpp' -or -name '*.hpp' -or -name 'CMakeLists.txt' \) | xargs chmod 0644 

        bash_lib_status "Done."
    fi
done

popd

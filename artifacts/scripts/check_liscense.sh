#!/bin/bash

# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

echo "Check MIT Liscense boilerplate..."
PWD="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
# TO source code root
pushd "${PWD}/../" > /dev/null

EXITCODE=0

declare -a BLACKLIST=()
filter=""
pre="-path '"
pos="' -prune -false -o "
for val in ${BLACKLIST[@]}; do
    filter+="${pre}""${val}""${pos}"
done

for SRC_FILE in $(find . ${filter} -type f \
    -and \( -name 'CMakeLists.txt' -or -name '*.cpp' -or -name '*.cu' -or -name '*.h'  -or -name '*.hpp' \
    -or -name '*.in' -or -name '*.py' -or -name '*.sh' -or -name '*.dockerfile' -or -name '*.yaml' \) ); do
    if !(grep -q "Copyright" "${SRC_FILE}"); then
        echo "[ERROR] Require: Liscense biolerplate" "${SRC_FILE}"
        EXITCODE=1
    fi
done

echo "Done."
popd > /dev/null
exit $EXITCODE
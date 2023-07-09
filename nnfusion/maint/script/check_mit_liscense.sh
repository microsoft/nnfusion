#!/bin/bash

# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

echo "Check MIT Liscense boilerplate..."
PWD="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
# TO source code root
pushd "${PWD}/../../" > /dev/null

EXITCODE=0

for SRC_FILE in $(find . -path './thirdparty' -prune -false -o -path './build' -prune -false -o -type f -not -name '*apply_mit_liscense.sh' \
    -not -name '*check_mit_liscense.sh' -and \( -name 'CMakeLists.txt' -or -name '*.cpp' -or -name '*.cu' -or -name '*.h'  -or -name '*.hpp' \
    -or -name '*.in' -or -name '*.py' -or -name '*.sh' -or -name '*.dockerfile' -or -name '*.yaml' \) ); do
    if !(grep -q "Copyright (c) Microsoft Corporation." "${SRC_FILE}") || !(grep -q "Licensed under the MIT License." "${SRC_FILE}") \
    || (grep -q "Copyright (c) Microsoft Corporation. All rights reserved." "${SRC_FILE}") \
    || (grep -q "Licensed under the MIT License. See License.txt in the project root for license information." "${SRC_FILE}") \
    || (grep -q -i -P "Microsoft( |)\(c\)" "${SRC_FILE}") || (grep -q "Apache License" "${SRC_FILE}"); then
        echo "[ERROR] Require: MIT Liscense biolerplate" "${SRC_FILE}"
        EXITCODE=1
    fi
done

echo "Done."
popd > /dev/null
exit $EXITCODE
#!/bin/bash

# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

echo "Check Apache2 Liscense boilerplate..."
PWD="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
# TO source code root
pushd "${PWD}/../../" > /dev/null

EXITCODE=0

for SRC_FILE in $(find . -type f -path './thirdparty/ngraph/*' -and \( -name '*.cpp' -or -name '*.hpp' \) ); do
    if !(grep -q "// Copyright 2017-2020 Intel Corporation" "${SRC_FILE}"); then
        echo "[ERROR] Require: Apache2 Liscense biolerplate" "${SRC_FILE}"
        EXITCODE=1
    fi
done

echo "Done."
popd > /dev/null
exit $EXITCODE
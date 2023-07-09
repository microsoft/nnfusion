#!/bin/bash

# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

CLANG_FORMAT="clang-format-3.9"
echo "Using ${CLANG_FORMAT} checking source code tree..."
PWD="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
# TO source code root
pushd "${PWD}/../../" > /dev/null

EXITCODE=0

for SRC_FILE in $(( find . -path './thirdparty' -prune -o -path './artifacts' -prune -o -path './build' -prune -o -path './src/tools/nnfusion/templates' -prune -o -type f -and \( -name '*.cpp' -or -name '*.hpp' \) ) | cat); do
    if "${CLANG_FORMAT}" -style=file -output-replacements-xml "${SRC_FILE}" 2>&1 | grep -v "Is a directory" | grep -c "<replacement " >/dev/null; then
        echo "[ERROR] Require: ${CLANG_FORMAT}" -style=file -i "${SRC_FILE}"
        EXITCODE=1
    fi
done

echo "Done."
popd > /dev/null
exit $EXITCODE
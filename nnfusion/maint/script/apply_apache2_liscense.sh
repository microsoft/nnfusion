#!/bin/bash

# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

echo "Add Apache2 Liscense boilerplate ..."
PWD="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
# TO source code root
pushd "${PWD}/../../" > /dev/null

EXITCODE=0

for SRC_FILE in $(find . -type f -path './thirdparty/ngraph/*' -and \( -name '*.cpp' -or -name '*.hpp' \) ); do
    sed -i 's/Copyright 2017-201[7-9]/Copyright 2017-2020/g' ${SRC_FILE}
    if !(grep -q "Copyright 2017-2020" "${SRC_FILE}"); then
        cat maint/script/apache2_liscense.txt ${SRC_FILE} > ${SRC_FILE}.new
        mv ${SRC_FILE}.new ${SRC_FILE}
    fi
done

echo "Done."
popd > /dev/null
exit $EXITCODE
#!/bin/bash

# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

declare THIS_SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

if [ -f "/.dockerenv" ]; then
  umask 000
fi

# do check code style
$THIS_SCRIPT_DIR/check_code_style.sh
if [ $? -ne 0 ]; then
    echo "Code style check failed. Please using apply_code_style.sh to apply code style."
    exit 1
else
    echo "Code style check succeded."
fi

# build the repo
mkdir -p $THIS_SCRIPT_DIR/../../build

pushd $THIS_SCRIPT_DIR/../../build > /dev/null
cmake ..
popd > /dev/null

if [ $? -ne 0 ]; then
    echo "CMake failed."
    exit 1
else
    echo "CMake succeded."
fi

# Make
cmake --build $THIS_SCRIPT_DIR/../../build -j$(nproc)

if [ $? -ne 0 ]; then
    echo "Build failed."
    exit 1
else
    echo "Build succeded."
fi

if [ -f "/.dockerenv" ]; then
    # Make install
    cmake --install $THIS_SCRIPT_DIR/../../build

    if [ $? -ne 0 ]; then
        echo "Install failed."
        exit 1
    else
        echo "Install succeded."
    fi
fi

pushd $THIS_SCRIPT_DIR/../../> /dev/null
rm src/python/nnfusion/nnfusion.tar.gz
rm -rf dist
python setup.py bdist_wheel
popd > /dev/null
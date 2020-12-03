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
if [ ! -d "$THIS_SCRIPT_DIR/../../build" ]; then
 mkdir $THIS_SCRIPT_DIR/../../build
fi

pushd $THIS_SCRIPT_DIR/../../build/ > /dev/null
cmake ..
popd > /dev/null

if [ $? -ne 0 ]; then
    echo "CMake failed."
    exit 1
else
    echo "CMake succeded."
fi

# Make
pushd $THIS_SCRIPT_DIR/../../build/ > /dev/null
make -j$(nproc)
popd > /dev/null

if [ $? -ne 0 ]; then
    echo "Build failed."
    exit 1
else
    echo "Build succeded."
fi

if [ -f "/.dockerenv" ]; then
    # Make install
    pushd $THIS_SCRIPT_DIR/../../build/ > /dev/null
    make install
    popd > /dev/null

    if [ $? -ne 0 ]; then
        echo "Install failed."
        exit 1
    else
        echo "Install succeded."
    fi
fi
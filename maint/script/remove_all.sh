#!/bin/bash

# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

declare THIS_SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

if [ -f "/.dockerenv" ]; then
    # clean product
    #rm /usr/local/bin/nnfusion
    #rm /usr/local/lib/libnnfusion_backend.so
    #rm /usr/local/lib/libnnfusion_operators.so
    #rm -rf /usr/local/bin/templates
    # clean build cache
    rm -rf $THIS_SCRIPT_DIR/../../build
    # clean everything
    rm -rf $THIS_SCRIPT_DIR/../../*
else
    docker exec -t nnfusion_base_dev /nnfusion/maint/script/remove_all.sh
fi
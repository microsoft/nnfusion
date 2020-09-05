#!/bin/bash

# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

declare THIS_SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Downlaod Models
$THIS_SCRIPT_DIR/download_models.sh

# Run building
$THIS_SCRIPT_DIR/build.sh

# Run test
python3 $THIS_SCRIPT_DIR/../../test/nnfusion/scripts/e2e_tests.py $THIS_SCRIPT_DIR/../../test/nnfusion/scripts/cpu_perf.json
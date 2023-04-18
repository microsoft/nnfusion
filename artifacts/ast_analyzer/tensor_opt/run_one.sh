#!/bin/bash
set -o pipefail
set -e
cd $1/nnfusion_rt/^^RT_DIR
timeout 25s ./main_test &> ../../run.log
if [ $? -ne 0 ]; then
    echo "timeout"
    exit 1
fi

#!/bin/bash

# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.


declare THIS_SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

echo "Build CUDA container"
docker build -t rammer/cuda:10.2-cudnn7-devel-ubuntu16.04 -f $THIS_SCRIPT_DIR/../dockerfile/nnfusion_cuda-10.2.dockerfile .
echo "Built CUDA image: rammer/cuda:10.2-cudnn7-devel-ubuntu16.04"
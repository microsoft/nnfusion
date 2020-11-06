#!/bin/bash

# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

declare THIS_SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

docker build -t nnfusion/ubuntu:18.04 -f $THIS_SCRIPT_DIR/../dockerfile/nnfusion_ubuntu_18.04.dockerfile .

res=`lspci | grep -i nvidia`
if [ ! -z "$res" ]; then
    echo "Build CUDA container"
    docker build -t nnfusion/cuda:10.2-cudnn7-devel-ubuntu18.04 -f $THIS_SCRIPT_DIR/../dockerfile/nnfusion_cuda-10.2.dockerfile .
fi

res=`lspci | grep -i amd/ati`
if [ ! -z "$res" ]; then
    echo "Build ROCm container"
    docker build -t nnfusion/rocm:3.5-ubuntu-18.04 -f $THIS_SCRIPT_DIR/../dockerfile/nnfusion_rocm-3.5.dockerfile .
fi
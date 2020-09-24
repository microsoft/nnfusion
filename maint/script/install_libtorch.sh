#!/bin/bash -e

# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

if [ ${Torch_DIR} ] && [ -d ${Torch_DIR} ]; then
    echo "- Found libtorch in ${Torch_DIR}, skipped installation"
    exit 0
fi

if [ ! -d "${HOME}/libtorch" ]; then
    echo "- Downloading libtorch"
    wget "https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-1.6.0%2Bcpu.zip"
    unzip -qq "libtorch-cxx11-abi-shared-with-deps-1.6.0+cpu.zip" -d ${HOME}
    rm libtorch-cxx11-abi-shared-with-deps-1.6.0+cpu.zip
fi

if [ "${Torch_DIR}" != "${HOME}/libtorch" ]; then
    if [[ "${SHELL}" != *"bash" ]]; then
        echo "- You might not use bash, please manually set Torch_DIR to ${HOME}/libtorch by yourself"
    else
        echo "export Torch_DIR=${HOME}/libtorch" >> ~/.bashrc
    fi
fi

echo "- Libtorch is installed in ${HOME}/libtorch, please restart shell to take effect"
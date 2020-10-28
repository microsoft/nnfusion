#!/bin/bash

# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

declare THIS_SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
declare MODELS=$THIS_SCRIPT_DIR/../../models

create_container(){
    docker kill $1 >/dev/null 2>&1 || true
    docker rm $1 >/dev/null 2>&1 || true
    docker run --name $1 -t -d --net=host -e EXEC_BASH=1 -v $THIS_SCRIPT_DIR/../../:/nnfusion -v $THIS_SCRIPT_DIR/../../../frozenmodels:/frozenmodels -w /nnfusion $2 bash
}

create_cuda_container(){
    docker kill $1 >/dev/null 2>&1 || true
    docker rm $1 >/dev/null 2>&1 || true
    docker run --gpus all --name $1 -t -d --net=host -e EXEC_BASH=1 -v $THIS_SCRIPT_DIR/../../:/nnfusion -v $THIS_SCRIPT_DIR/../../../frozenmodels:/frozenmodels -w /nnfusion $2 bash
}

create_rocm_container(){
    docker kill $1 >/dev/null 2>&1 || true
    docker rm $1 >/dev/null 2>&1 || true
    docker run --name $1 -t -d --net=host --privileged --device=/dev/kfd --device=/dev/dri --security-opt seccomp=unconfined --group-add video -e EXEC_BASH=1 -v $THIS_SCRIPT_DIR/../../:/nnfusion -v $THIS_SCRIPT_DIR/../../../frozenmodels:/frozenmodels -w /nnfusion $2 bash
}

# check if inside one docker container(for testing)
if [ -f "/.dockerenv" ]; then
    $THIS_SCRIPT_DIR/build.sh
    python3 $THIS_SCRIPT_DIR/../../test/nnfusion/scripts/e2e_tests.py
else
    
    if [ ! -d "$THIS_SCRIPT_DIR/../../models/frozenmodels/" ]; then
        # prepare models
        if [ ! -d "$THIS_SCRIPT_DIR/../../../frozenmodels/" ]; then
            $THIS_SCRIPT_DIR/download_models.sh
        fi
    fi

    # use nnfusion_base for build / test cpu
    create_container nnfusion_base_dev nnfusion/ubuntu:18.04
    if [ $? -ne 0 ]; then
        echo "One or many Docker containers were not built. Run ./build_containers.sh ."
        exit 1
    fi

    # build & install
    docker exec -t nnfusion_base_dev /nnfusion/maint/script/build.sh

    if [ $? -ne 0 ]; then
        docker exec -t nnfusion_base_dev sh -c "rm -rf /nnfusion/build"
        exit 1
    fi

    failed=0

    echo "Launch BASE/Generic_cpu container to test:"
    docker exec -t nnfusion_base_dev /nnfusion/maint/script/test.sh &

    res=`lspci | grep -i amd/ati`
    if [ ! -z "$res" ]; then
        echo "Launch ROCm container to test:"
        create_rocm_container nnfusion_rocm_dev nnfusion/rocm:3.5-ubuntu-18.04 && docker exec -t nnfusion_rocm_dev /nnfusion/maint/script/test.sh &
    fi

    res=`lspci | grep -i nvidia`
    if [ ! -z "$res" ]; then
        echo "Launch Cuda container to test:"
        create_cuda_container nnfusion_cuda_dev nnfusion/cuda:10.2-cudnn7-devel-ubuntu18.04 && docker exec -t nnfusion_cuda_dev /nnfusion/maint/script/test.sh &
    fi

    # HLSL Docker?
    # docker exec -t nnfusion_hlsl_dev /source/maint/script/test.sh  --direct

    for job in `jobs -p`
    do
        wait $job || let "failed+=1"
    done

    docker exec -t nnfusion_base_dev sh -c "rm -rf /nnfusion/build /nnfusion/nnfusion_rt"
    docker exec -t nnfusion_base_dev sh -c "find /nnfusion -type d -name '__pycache__' | xargs rm -rf"

    if [ ! "$failed" == "0" ]; then
        echo "Test execution failed."
        exit 1
    fi

fi

#!/bin/bash

# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

declare THIS_SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
declare MODELS=$THIS_SCRIPT_DIR/../../models

create_cuda_container(){
    docker kill $1 >/dev/null 2>&1 || true
    docker rm $1 >/dev/null 2>&1 || true
    docker run --gpus all --name $1 -t -d --net=host -e EXEC_BASH=1 -v $THIS_SCRIPT_DIR/../../:/nnfusion -v $THIS_SCRIPT_DIR/../../../frozenmodels:/frozenmodels -w /nnfusion $2 bash
}

# check if inside one docker container(for testing)
if [ -f "/.dockerenv" ]; then
    $THIS_SCRIPT_DIR/build.sh
    python3 $THIS_SCRIPT_DIR/../../test/nnfusion/scripts/e2e_tests.py $THIS_SCRIPT_DIR/../../test/nnfusion/scripts/perf.json
else
    
    if [ ! -d "$THIS_SCRIPT_DIR/../../models/frozenmodels/" ]; then
        # prepare models
        if [ ! -d "$THIS_SCRIPT_DIR/../../../frozenmodels/" ]; then
            $THIS_SCRIPT_DIR/download_models.sh
        fi
    fi

    # use nnfusion_base for build / test cpu
    create_cuda_container nnfusion_cuda_dev nnfusion/cuda:10.2-cudnn7-devel-ubuntu18.04
    if [ $? -ne 0 ]; then
        echo "One or many Docker containers were not built. Run ./build_containers.sh ."
        exit 1
    fi

    # build & install
    # docker exec -t nnfusion_cuda_dev /nnfusion/maint/script/build.sh

    failed=0

    res=`lspci | grep -i nvidia`
    if [ ! -z "$res" ]; then
        echo "Launch Cuda container to benchmark:"
        docker exec -t nnfusion_cuda_dev /nnfusion/maint/script/benchmark.sh &
    fi

    for job in `jobs -p`
    do
        wait $job || let "failed+=1"
    done

    docker exec -t nnfusion_cuda_dev sh -c "rm -rf /nnfusion/build /nnfusion/nnfusion_rt"
    docker exec -t nnfusion_cuda_dev sh -c "find /nnfusion -type d -name '__pycache__' | xargs rm -rf"

    if [ ! "$failed" == "0" ]; then
        echo "Test execution failed."
        exit 1
    fi

fi

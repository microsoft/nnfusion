#!/bin/bash -e

# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

echo "     _  __ _  __ ____            _ "
echo "    / |/ // |/ // __/__ __ ___  (_)___   ___ "
echo "   /    //    // _/ / // /(_-< / // _ \\ / _ \\"
echo "  /_/|_//_/|_//_/   \\_,_//___//_/ \\___//_//_/"
echo "      MSRAsia NNFusion Team(@nnfusion)"
echo "              #roller_cuda"

declare DOCKERNAME=roller_cuda
declare THIS_SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
check_results=`docker ps --filter "name=${DOCKERNAME}"`

if [[ "$check_results" == *"$DOCKERNAME"* ]] 
then
    docker start $DOCKERNAME > /dev/null 2>&1
    docker exec -it $DOCKERNAME bash
else
    docker run --gpus=all --name $DOCKERNAME -it -d --net=host -e EXEC_BASH=1 -v $THIS_SCRIPT_DIR/../../:/nnfusion -w /root/nnfusion roller/cuda:10.2-cudnn7-devel-ubuntu16.04 bash 
    docker exec -it $DOCKERNAME bash
fi

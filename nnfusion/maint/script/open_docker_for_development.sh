#!/bin/bash -e

# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

echo "     _  __ _  __ ____            _ "
echo "    / |/ // |/ // __/__ __ ___  (_)___   ___ "
echo "   /    //    // _/ / // /(_-< / // _ \\ / _ \\"
echo "  /_/|_//_/|_//_/   \\_,_//___//_/ \\___//_//_/"
echo "      MSRAsia NNFusion Team(@nnfusion)"
echo ""

declare DOCKERNAME=nnf_cpu_dev
declare THIS_SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
check_results=`docker ps --filter "name=${DOCKERNAME}"`

if [[ "$check_results" == *"$DOCKERNAME"* ]] 
then
    docker start $DOCKERNAME > /dev/null 2>&1
    docker exec -it $DOCKERNAME bash
else
    echo "Running create_a_docker_for_development.sh"
    echo "- Make sure docker installed on system"
    echo "- The development enviroment is based on Ubuntu 18.04."
    echo "- A container named nnfusion will be created or reloaded."

    #docker kill nnfusion_dev >/dev/null 2>&1 || true
    #docker rm nnfusion_dev >/dev/null 2>&1 || true
    docker run --name $DOCKERNAME -it -d --net=host -e EXEC_BASH=1 -v $THIS_SCRIPT_DIR/../../:/nnfusion -w /nnfusion ubuntu:18.04 bash > /dev/null 2>&1
    docker exec -t $DOCKERNAME /nnfusion/maint/script/install_dependency.sh
    docker exec -it $DOCKERNAME bash
fi

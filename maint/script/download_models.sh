#!/bin/bash

# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

declare THIS_SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
declare MODELS=$THIS_SCRIPT_DIR/../../
declare NNF_FILE_SERV="10.190.174.54"

if ping -c 1 $NNF_FILE_SERV &> /dev/null
then
    #sync files
    rsync -a nnfusion@$NNF_FILE_SERV:/home/nnfusion/pipeline/models $MODELS
    if [ $? -ne 0 ]; then
        echo "Download models failed."
        exit 1
    else
        echo "Download models succeded."
    fi
else
    echo "NNFusion fileserver:$NNF_FILE_SERV is not reachable;"
    exit 2
fi
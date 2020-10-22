#!/bin/bash

# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

declare THIS_SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
declare MODELS=$THIS_SCRIPT_DIR/../../models/frozenmodels/
# Please use valid sas_token here
if [ -z $SAS_TOKEN];then
    echo "Please specify a valid SAS_TOKEN for uploading to azure storage service;"
    exit 1
fi

# install az cli
if which az >/dev/null; then
    echo "az existed"
else
    echo "installing az ..."
    curl -sL https://aka.ms/InstallAzureCLIDeb | sudo bash
fi

# upload pipline folder
for subfolder in `ls $MODELS`
do
    subdir=$MODELS"/"$subfolder
    if [ -d $subdir ]
    then 
        echo "az uploading $subdir..."
        az storage copy -s $subdir -d https://nnfusion.blob.core.windows.net/frozenmodels --recursive --sas-token $SAS_TOKEN
    fi
done
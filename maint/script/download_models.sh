#!/bin/bash

# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

declare THIS_SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
declare MODELS=$THIS_SCRIPT_DIR/../../../
declare SAS_TOKEN="?sv=2019-12-12&ss=bfqt&srt=co&sp=rl&se=2021-10-22T01:39:45Z&st=2020-10-21T17:39:45Z&spr=https&sig=KH9GB6LO7hguUPH5%2BdF8YKJk%2By55cVQlSHmExOT1uAg%3D"

# install az cli
if which az >/dev/null; then
    echo "az existed"
else
    echo "installing az ..."
    curl -sL https://aka.ms/InstallAzureCLIDeb | sudo bash
fi

# download all the files & save to same level as source code
az storage copy -s https://nnfusion.blob.core.windows.net/frozenmodels -d $MODELS --recursive --sas-token $SAS_TOKEN
rm -rf $THIS_SCRIPT_DIR/../../models/frozenmodels
ln -s $THIS_SCRIPT_DIR/../../../frozenmodels/  $THIS_SCRIPT_DIR/../../models/frozenmodels
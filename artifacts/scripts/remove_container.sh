# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

docker kill rammer_cuda>/dev/null 2>&1 || true
docker rm rammer_cuda >/dev/null 2>&1 || true
echo "Removed rammer_cuda."

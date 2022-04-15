# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

docker kill roller_cuda>/dev/null 2>&1 || true
docker rm roller_cuda >/dev/null 2>&1 || true
echo "Removed roller_cuda."

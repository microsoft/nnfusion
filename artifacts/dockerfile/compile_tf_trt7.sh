# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

bazel build --cxxopt="-D_GLIBCXX_USE_CXX11_ABI=0" --config=v1 --config=opt --config=cuda //tensorflow/tools/pip_package:build_pip_package

bazel-bin/tensorflow/tools/pip_package/build_pip_package /tmp/tensorflow_pkg

cp /tmp/tensorflow_pkg/tensorflow-1.15.2-cp36-cp36m-linux_x86_64.whl ./
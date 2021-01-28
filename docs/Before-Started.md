Below figures show the required dependencies, as well as recommended versions of operating systems, runtime and supported hardware.

Please make sure the lib version or OS version is correct. 

1. Recommended OS: Ubuntu 16.04 or Ubuntu 18.04
2. Package&Dependency

*Note:* 

* *If you use native system to build NNFusion, we will provide [install_dependency_script](https://github.com/microsoft/nnfusion/blob/master/maint/script/install_dependency.sh) to install all the dependencies.* 
* *If you use docker to build NNFusion, we will wrap all the dependencies in the image.*

| Package/Dependency   | Verified Version on Ubuntu 18.04 | Verified Version on Ubuntu 16.04 |
| -------------------- | -------------------------------- | -------------------------------- |
| build-essential      | 12.4                             | 12.1                             |
| CMake                | 3.18.4                           | 3.18.4                           |
| clang                | 3.9.1                            | 3.9.1                            |
| clang-format         | 3.9.1                            | 3.9.1                            |
| git                  | 2.17.1                           | 2.7.4                            |
| curl                 | 7.58                             | 7.47.0                           |
| zilib1g              | 1.2.11                           | 1.2.8                            |
| zilib1g-dev          | 1.2.11                           | 1.2.8                            |
| libtinfo-dev         | 6.1                              | 6                                |
| unzip                | 6                                | 6                                |
| autoconf             | 2.69                             | 2.69                             |
| automake             | 1.51.1                           | 1.15                             |
| libtool              | 2.4.6                            | 2.4.6                            |
| ca-certificates      | 20190110~18.04.1                 | 20190110~16.04.1                 |
| gcc                  | 7.5.0                            | 5.4.0                            |
| g++                  | 7.5.0                            | 5.4.0                            |
| gdb                  | 8.1                              | 7.11.1                           |
| sqlite3              | 3.22                             | 3.11.0                           |
| libsqlite3-dev       | 3.22                             | 3.11.0                           |
| libcurl4-openssl-dev | 7.58.0                           | 7.47.0                           |
| libprotobuf-dev      | 3.6.1                            | 3.6.1                            |
| protobuf-compiler    | 3.6.1                            | 3.6.1                            |
| libgflags-dev        | 2.2.1                            | 2.1.2                            |
| libgtest-dev         | 1.9.0                            | 1.9.0                            |

3. Runtime

| Runtime | Verified Version on Ubuntu 18.04 | Verified Version on Ubuntu 16.04 |
| ------- | -------------------------------- | -------------------------------- |
| CUDA    | 10                               | NA                               |
| ROCM    | 3.5                              | NA                               |

4. The recommended Tensorflow version is 1.14 and below.
5. Supported GPU cards

| Supported GPU cards |
| ------------------- |
| Nvidia V100         |
| Nvidia P100         |
| Nvidia 1080         |
| Nvidia 1080ti       |
| Nvidia 2080ti       |
| AMD Vega 20         |

After you check all the dependencies, versions of OS and etc. You are ready to use NNFusion to compile your model.
NNFusion support models of pb and ONNX format. Please refer to [Freeze TensorFlow Model](https://github.com/microsoft/nnfusion/blob/master/docs/Freeze-TensorFlow-Models.md) and [Freeze PyTorch Model](https://github.com/microsoft/nnfusion/blob/master/docs/Freeze-PyTorch-Model.md).

Refer to [Build Guide](https://github.com/microsoft/nnfusion/blob/master/docs/Build-Guide.md) to learn how to compile and install NNFusion.

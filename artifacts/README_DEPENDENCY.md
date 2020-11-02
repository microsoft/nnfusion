# Install dependencies and Rammer

1. clang-llvm-6.0.0 (for TVM, ready in .deps)
2. cmake-3.18.0 (for Rammer, ready in .deps)
3. cuda-10.0, cuDNN-7.6.5 (ready in .deps)
4. TensorRT-7.0.0.11 (baseline in evaluation) (ready in .deps)
5. bazel-0.26.1 (for TensorFlow-1.15.2, ready in .deps)
6. Python runtime: Anaconda3
7. TensorFlow-GPU-1.14.0 (for model frozen)
8. tflearn (for model frozen)
9. gast-0.2.2 (for model frozen)
9. TensorFlow-GPU-1.15.2 (baseline in evaluation)
10. TVM-0.7 (commit 24c53a343b0ecb76ed766d3f29e968ee0f8b0816) (baseline in evaluation)
11. Rammer (NNFusion)
12. numpy-1.16.4 (fix TensorFlow warning: https://github.com/tensorflow/tensorflow/issues/31249#issuecomment-517484918)

## 6. Install Anaconda3 as the Python runtime

```bash
Download Anaconda3-5.1.0-Linux-x86_64.sh into .deps/
cd .deps/
bash ./Anaconda3-5.1.0-Linux-x86_64.sh -b -p ../.deps/anaconda3
cd ..
```

## 7. Install the official pre-built TensorFlow-1.14.0 for model frozen

```bash
pip install tensorflow-gpu==1.14.0
```

After freezing models, this TensorFlow-1.14.0 should be uninstalled.

## 8. Install tflearn

```bash
pip install tflearn
```

## 9. Build and install TensorFlow-1.15.2 with TensorRT-7.0 support

The official pre-built TensorFlow-1.15.2 only supports TensorRT-5.0, so we need to build TensorFlow from source to support TensorRT-7.0.

This version should be installed after uninstalling TensorFlow-1.14.0.

### Install the pre-built whl

```bash
pip uninstall tensorflow-gpu -y # uninstall tensorflow-1.14.0
pip install .deps/tensorflow-trt/tensorflow-1.15.2-cp36-cp36m-linux_x86_64.whl
```

### or build from source

```bash
pip uninstall tensorflow-gpu -y # uninstall tensorflow-1.14.0
cd .deps/tensorflow-trt # see clone tensorflow-trt
pip install keras_preprocessing
./configure # see configure.screensnap
bash compile.sh
pip install tensorflow-1.15.2-cp36-cp36m-linux_x86_64.whl
cd ../..
```

Note that if you modify the folder path, you need to build tensorflow-trt from source.

### clone tensorflow-trt (done)

```bash
cd .deps/
git clone https://github.com/tensorflow/tensorflow.git tensorflow-trt
cd tensorflow-trt
git checkout v1.15.2
cd ..
```

### configure.screensnap

```bash
lingm@nni-gpu-01:~/artifacts/.deps/tensorflow-trt$ ./configure
WARNING: --batch mode is deprecated. Please instead explicitly shut down your Bazel server using the command "bazel shutdown".
You have bazel 0.26.1 installed.
Please specify the location of python. [Default is /home/lingm/anaconda3/bin/python]:


Found possible Python library paths:
  /home/lingm/anaconda3/lib/python3.6/site-packages
Please input the desired Python library path to use.  Default is [/home/lingm/anaconda3/lib/python3.6/site-packages]

Do you wish to build TensorFlow with XLA JIT support? [Y/n]:
XLA JIT support will be enabled for TensorFlow.

Do you wish to build TensorFlow with OpenCL SYCL support? [y/N]:
No OpenCL SYCL support will be enabled for TensorFlow.

Do you wish to build TensorFlow with ROCm support? [y/N]:
No ROCm support will be enabled for TensorFlow.

Do you wish to build TensorFlow with CUDA support? [y/N]: y
CUDA support will be enabled for TensorFlow.

Do you wish to build TensorFlow with TensorRT support? [y/N]: y
TensorRT support will be enabled for TensorFlow.

Could not find any NvInferVersion.h matching version '' in any subdirectory:
        ''
        'include'
        'include/cuda'
        'include/*-linux-gnu'
        'extras/CUPTI/include'
        'include/cuda/CUPTI'
of:
        '/lib'
        '/lib/x86_64-linux-gnu'
        '/lib32'
        '/usr'
        '/usr/lib'
        '/usr/lib/x86_64-linux-gnu'
        '/usr/lib/x86_64-linux-gnu/libfakeroot'
        '/usr/lib32'
        '/usr/local/cuda'
        '/usr/local/cuda-10.2/targets/x86_64-linux/lib'
        '/usr/local/cuda/lib64'
        '/usr/local/lib'
Asking for detailed CUDA configuration...

Please specify the CUDA SDK version you want to use. [Leave empty to default to CUDA 10]:


Please specify the cuDNN version you want to use. [Leave empty to default to cuDNN 7]:


Please specify the TensorRT version you want to use. [Leave empty to  default to TensorRT 5]: 7


Please specify the locally installed NCCL version you want to use. [Leave empty to use http://github.com/nvidia/nccl]:


Please specify the comma-separated list of base paths to look for CUDA libraries and headers. [Leave empty to use the default]: /home/lingm/.local,/home/lingm/.local/cuda-10.0,/home/lingm/.local/cuda-10.0/bin,/home/lingm/.local/cuda-10.0/include,/home/lingm/.local/cuda-10.0/lib64,/home/lingm/.local/TensorRT-7.0.0.11/include,/home/lingm/.local/TensorRT-7.0.0.11/lib


Found CUDA 10.0 in:
    /home/lingm/.local/cuda-10.0/lib64
    /home/lingm/.local/cuda-10.0/include
Found cuDNN 7 in:
    /home/lingm/.local/cuda-10.0/lib64
    /home/lingm/.local/cuda-10.0/include
Found TensorRT 7 in:
    /home/lingm/.local/TensorRT-7.0.0.11/lib
    /home/lingm/.local/TensorRT-7.0.0.11/include


Please specify a list of comma-separated CUDA compute capabilities you want to build with.
You can find the compute capability of your device at: https://developer.nvidia.com/cuda-gpus.
Please note that each additional compute capability significantly increases your build time and binary size, and that TensorFlow only supports compute capabilities >= 3.5 [Default is: 7.0]:


Do you want to use clang as CUDA compiler? [y/N]:
nvcc will be used as CUDA compiler.

Please specify which gcc should be used by nvcc as the host compiler. [Default is /usr/bin/gcc]:


Do you wish to build TensorFlow with MPI support? [y/N]:
No MPI support will be enabled for TensorFlow.

Please specify optimization flags to use during compilation when bazel option "--config=opt" is specified [Default is -march=native -Wno-sign-compare]:


Would you like to interactively configure ./WORKSPACE for Android builds? [y/N]:
Not configuring the WORKSPACE for Android builds.

Preconfigured Bazel build configs. You can use any of the below by adding "--config=<>" to your build command. See .bazelrc for more details.
        --config=mkl            # Build with MKL support.
        --config=monolithic     # Config for mostly static monolithic build.
        --config=gdr            # Build with GDR support.
        --config=verbs          # Build with libverbs support.
        --config=ngraph         # Build with Intel nGraph support.
        --config=numa           # Build with NUMA support.
        --config=dynamic_kernels        # (Experimental) Build kernels into separate shared objects.
        --config=v2             # Build TensorFlow 2.x instead of 1.x.
Preconfigured Bazel build configs to DISABLE default on features:
        --config=noaws          # Disable AWS S3 filesystem support.
        --config=nogcp          # Disable GCP support.
        --config=nohdfs         # Disable HDFS support.
        --config=noignite       # Disable Apache Ignite support.
        --config=nokafka        # Disable Apache Kafka support.
        --config=nonccl         # Disable NVIDIA NCCL support.
Configuration finished
```

Note that in ```Please specify the comma-separated list of base paths to look for CUDA libraries and headers. [Leave empty to use the default]```, you need to input the path of CUDA (```/home/lingm/.local,/home/lingm/.local/cuda-10.0,/home/lingm/.local/cuda-10.0/bin,/home/lingm/.local/cuda-10.0/include,/home/lingm/.local/cuda-10.0/lib64```) and TensorRT (```/home/lingm/.local/TensorRT-7.0.0.11/include,/home/lingm/.local/TensorRT-7.0.0.11/lib```). Note that the path should not be a soft-link.

## 10. Build and install TVM-0.7

```bash
source .profile

sudo apt-get update
sudo apt-get install -y gcc libtinfo-dev zlib1g-dev build-essential cmake libedit-dev libxml2-dev

cd .deps/
git clone https://github.com/apache/incubator-tvm.git tvm-0.7
cd tvm-0.7
git checkout 24c53a343b0ecb76ed766d3f29e968ee0f8b0816 # this commit is used in the evaluation
git submodule init
git submodule update
mkdir build
cp ../../tvm-config.cmake ./config.cmake
cmake ..
make -j

apt-get install antlr4
# the latest version may make TVM slow
pip install tornado==4.5.3 psutil==5.4.3 xgboost==0.90 decorator==4.2.1 attrs==17.4.0 mypy==0.720 orderedset==2.0.1 antlr4-python3-runtime==4.7.2
```

### config.cmake screensnap

```bash
#--------------------------------------------------------------------
#  Template custom cmake configuration for compiling
#
#  This file is used to override the build options in build.
#  If you want to change the configuration, please use the following
#  steps. Assume you are on the root directory. First copy the this
#  file so that any local changes will be ignored by git
#
#  $ mkdir build
#  $ cp cmake/config.cmake build
#
#  Next modify the according entries, and then compile by
#
#  $ cd build
#  $ cmake ..
#
#  Then buld in parallel with 8 threads
#
#  $ make -j8
#--------------------------------------------------------------------

#---------------------------------------------
# Backend runtimes.
#---------------------------------------------

# Whether enable CUDA during compile,
#
# Possible values:
# - ON: enable CUDA with cmake's auto search
# - OFF: disable CUDA
# - /path/to/cuda: use specific path to cuda toolkit
set(USE_CUDA ON)

# Whether enable ROCM runtime
#
# Possible values:
# - ON: enable ROCM with cmake's auto search
# - OFF: disable ROCM
# - /path/to/rocm: use specific path to rocm
set(USE_ROCM OFF)

# Whether enable SDAccel runtime
set(USE_SDACCEL OFF)

# Whether enable Intel FPGA SDK for OpenCL (AOCL) runtime
set(USE_AOCL OFF)

# Whether enable OpenCL runtime
set(USE_OPENCL OFF)

# Whether enable Metal runtime
set(USE_METAL OFF)

# Whether enable Vulkan runtime
#
# Possible values:
# - ON: enable Vulkan with cmake's auto search
# - OFF: disable vulkan
# - /path/to/vulkan-sdk: use specific path to vulkan-sdk
set(USE_VULKAN OFF)

# Whether enable OpenGL runtime
set(USE_OPENGL OFF)

# Whether enable MicroTVM runtime
set(USE_MICRO OFF)

# Whether to enable SGX runtime
#
# Possible values for USE_SGX:
# - /path/to/sgxsdk: path to Intel SGX SDK
# - OFF: disable SGX
#
# SGX_MODE := HW|SIM
set(USE_SGX OFF)
set(SGX_MODE "SIM")
set(RUST_SGX_SDK "/path/to/rust-sgx-sdk")

# Whether enable RPC runtime
set(USE_RPC ON)

# Whether embed stackvm into the runtime
set(USE_STACKVM_RUNTIME OFF)

# Whether enable tiny embedded graph runtime.
set(USE_GRAPH_RUNTIME ON)

# Whether enable additional graph debug functions
set(USE_GRAPH_RUNTIME_DEBUG OFF)

# Whether build with LLVM support
# Requires LLVM version >= 4.0
#
# Possible values:
# - ON: enable llvm with cmake's find search
# - OFF: disable llvm
# - /path/to/llvm-config: enable specific LLVM when multiple llvm-dev is available.
set(USE_LLVM ON)

#---------------------------------------------
# Contrib libraries
#---------------------------------------------
# Whether use BLAS, choices: openblas, mkl, atlas, apple
set(USE_BLAS none)

# /path/to/mkl: mkl root path when use mkl blas library
# set(USE_MKL_PATH /opt/intel/mkl) for UNIX
# set(USE_MKL_PATH ../IntelSWTools/compilers_and_libraries_2018/windows/mkl) for WIN32
set(USE_MKL_PATH none)

# Whether use contrib.random in runtime
set(USE_RANDOM OFF)

# Whether use NNPack
set(USE_NNPACK OFF)

# Whether use CuDNN
set(USE_CUDNN OFF)

# Whether use cuBLAS
set(USE_CUBLAS OFF)

# Whether use MIOpen
set(USE_MIOPEN OFF)

# Whether use MPS
set(USE_MPS OFF)

# Whether use rocBlas
set(USE_ROCBLAS OFF)

# Whether use contrib sort
set(USE_SORT ON)

# Build ANTLR parser for Relay text format
set(USE_ANTLR OFF)

# Whether use Relay debug mode
set(USE_RELAY_DEBUG OFF)
```

## 11. Build and install Rammer (NNFusion)

```bash
clone nnfusion into rammer folder
cd rammer
mkdir build
cd build
cmake ..
make -j
make install 
cd ../..
```

## 12. numpy-1.16.4

To fix the warning message of TensorFlow ```FutureWarning: Passing (type, 1) or '1type' as a synonym of type is deprecated; in a future version of numpy, it will be understood as (type, (1,)) / '(1,)type'.```

Note that this warning message is harmless and can also be ignored.

```bash
pip install numpy==1.16.4
```
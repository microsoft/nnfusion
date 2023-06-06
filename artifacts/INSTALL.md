# Installation Tutorial
This document describes how to install the software used in the artifact on a node with NVIDIA GPU. All scripts are assumed to be run from `nnfusion/artifacts` directory.

## Prerequirements
We assume that you have a node with NVIDIA driver and CUDA installed. We also assume that you have installed conda and nvcc. If you have not installed conda, you can install it by following the instructions [here](https://docs.conda.io/projects/conda/en/latest/user-guide/install/linux.html) (Miniconda is enough, and this artifact assumes that miniconda is installed at the default path `~/miniconda3`). If you have not installed nvcc, you can install it by following the instructions [here](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html).

## TensorFlow
The onnx-tf for TF 1.15 needs to be built from source because the pre-compiled version depends on TF2. We also fix some bugs in that commit to properly support the control flow operations. The following commands will prepare the conda env for TF 1.15.

```bash
conda create python=3.8 --name baseline_tf1 -y
conda activate baseline_tf1
pip install nvidia-pyindex
pip install -r env/requirements_tf.txt
mkdir -p third-party && cd third-party
git clone https://github.com/onnx/onnx-tensorflow.git
cd onnx-tensorflow
git checkout 0e4f4836 # v1.7.0-tf-1.15m
git apply ../../env/onnx_tf.patch
pip install -e .
conda deactivate
```
## JAX
The following commands will prepare the conda env for JAX.
```bash
conda create python=3.8 --name baseline_jax -y
conda activate baseline_jax
pip install nvidia-pyindex
pip install -r env/requirements_jax.txt -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html -f https://download.pytorch.org/whl/torch_stable.html
conda deactivate
```

## TVM
The following commands will prepare the conda env for TVM.
```bash
conda create python==3.8 --name kerneldb -y
conda activate kerneldb
pip install ply==3.11
mkdir -p third-party && cd third-party
git clone https://github.com/apache/tvm.git
cd tvm
git checkout 22ba6523c
git submodule init && git submodule update
git apply ../../env/tvm.patch # from roller
mkdir build
cd build
cp ../cmake/config.cmake config.cmake
sed -i "s/USE_CUDA OFF/USE_CUDA ON/g" config.cmake && sed -i "s/USE_LLVM OFF/USE_LLVM ON/g" config.cmake
cmake .. && make -j
cd ../python
pip install -e .
```

## NNFusion
The following commands will build nnfusion. Please use the [script](../maint/script/install_dependency.sh) (needs sudo) to prepare the environment for nnfusion before running the following commands.

```bash
cd .. # to $YOUR_DIR_FOR_NNFUSION/nnfusion
mkdir build && cd build && cmake .. && make -j
```

## Pytorch & Cocktailer
```bash
conda create python=3.7 --name controlflow -y
conda activate controlflow
pip install nvidia-pyindex
pip install -r env/requirements_pytorch.txt -f https://download.pytorch.org/whl/torch_stable.html
pip install -e .
conda deactivate
```

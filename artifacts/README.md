# Installation of Evaluated Systems
assume running at artifacts directory


## Pre-requisites
conda, nvcc ......

## TensorFlow
install from env/requirements_tf.txt
Install onnx-tf from source (the pre-compiled version depends on TF2)

```bash
conda create python=3.8 --name baseline_tf1 -y
conda activate baseline_tf1
pip install nvidia-pyindex
pip install -r env/requirements_tf.txt
mkdir third-party && cd third-party
git clone https://github.com/onnx/onnx-tensorflow.git
cd onnx-tensorflow
git checkout 0e4f4836 # v1.7.0-tf-1.15m
git apply ../../env/onnx_tf.patch
pip install -e .
conda deactivate
```
## JAX
```bash
conda create python=3.8 --name baseline_jax -y
conda activate baseline_jax
pip install nvidia-pyindex
pip install -r env/requirements_jax.txt -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html -f https://download.pytorch.org/whl/torch_stable.html
conda deactivate
```

# TVM
```bash
conda create python==3.8 --name kerneldb -y
pip install ply==3.11
mkdir third-party && cd third-party
git clone https://github.com/apache/tvm.git --recursive
cd tvm
git checkout 22ba6523c
git apply ../../env/tvm.patch
mkdir build
cd build
cp ../../../env/tvm.config.cmake config.cmake
make -j
cd ../python
pip install -e .
```

## NNFusion

## Pytorch & Grinder
```bash
conda create python=3.7 --name grinder -y
conda activate grinder
pip install nvidia-pyindex
pip install -r env/requirements_pytorch.txt -f https://download.pytorch.org/whl/torch_stable.html
conda deactivate
```

## Grinder (with code)
```bash
export ARTIFACT_ROOT=***/ControlFlow/artifacts TODO
cd $ARTIFACT_ROOT/..
pip install -e .
```
TODO install nnfusion
TODO prepare kerneldb


docker: --shm-size="32g"
docker build -t grinder:latest -f env/Dockerfile.rocm --network=host .

cmake ..
```
cd $ARTIFACT_ROOT/../nnfusion
mkdir build && cd build
cmake .. && make -j
cd $ARTIFACT_ROOT/..
pip install -e . 
TODO: config.py
```

# build jax docker
```bash
mkdir third-party && cd third-party
git clone https://github.com/google/jax.git
cd jax
git checkout 0282b4bfad
git apply ../../env/jax.rocm.patch
./build/rocm/ci_build.sh --keep_image bash -c "./build/rocm/build_rocm.sh"
```


srun --pty -w nico3 -p Long --exclusive ./run_nv_gpu.sh 

cd plot && ./plot_nv.sh && cd -
# Generate VGG16 TorchScript model

## Build NNFusion with TorchScript support

By default PyTorch is installed with pre-cxx11 ABI, which is incompatible with NNFusion. You could download cxx11 ABI version by:

```bash
wget https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-1.6.0%2Bcpu.zip
unzip libtorch-cxx11-abi-shared-with-deps-1.6.0+cpu.zip
```

Then we could go back to NNFusion, compile it with TorchScript supports:

```bash
mkdir build && cd build
Torch_DIR=/path/to/libtorch/folder cmake .. -DTORCHSCRIPT_FRONTEND=1
make -j
```

## Compile TorchScript VGG16

```bash
## Generate VGG16 trace model and print its output with ones inputs
python main.py

## Codegen with specific input shape/type
./src/tools/nnfusion/nnfusion /path/to/vgg16_trace_module.pt -f torchscript -p 1,3,224,224:float
```

## Test compiled model

Compared NNFusion result against TorchScript

```bash
## By default it generates cuda code
cd nnfusion_rt/cuda_codegen
cmake .
make -j
./main_test
```

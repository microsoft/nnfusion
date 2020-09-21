## Generate VGG16 trace model


## Build NNFusion with TorchScript support
PyTorch is installed with pre-cxx11 ABI, which is incompatible with NNFusion. Luckily, PyTorch team provides cxx11 ABI version:
```bash
wget https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-1.6.0%2Bcpu.zip
unzip libtorch-cxx11-abi-shared-with-deps-1.6.0+cpu.zip
```
Then we could go back to project folder, compile NNFusion with TorchScript:
```bash
mkdir build && cd build
cmake .. -DTORCHSCRIPT_FRONTEND=1 -DTORCH_PREFIX=/path/to/libtorch/folder
make -j
```

## Compile TorchScript VGG16
```bash
## Generate VGG16 trace model and print its output
python main.py

## Compile model with shape/type
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




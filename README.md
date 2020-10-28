## Introduction
NNFusion(Nerual Network Fusion) is a flexible and efficient DNN compiler that can generate high-performance executables from a DNN model description. NNFusion is designed to facilitate DNN compiler research, with full stack of optimizations built-in and can target different accelerator devices.

## Who should consider using NNFusion?
- Those who want to use different accelerator device to optimize model compilation.
- End users who want to generate high-performance executables from DNN model description of different frameworks(ONNX,TensorFlow and PyTorch).
- Researchers who want to optimize model compilation and apply the method to their own work.
- Contributors who want to do DNN compiler optimization research and are willing to contribute.

## Highlight features
- Source-to-source compilation to remove framework overhead and dependency
- Full stack of optimization passes built-in
- Compile-time kernel co-scheduling abstraction for general accelerator devices
- Tight integration and co-optimization with communication, data loading and job scheduling
- Automatic kernel tuning through interacting with kernel tuning service
- Compatible with both hand-crafted kernel code and vendor-provided libraries
- Customized optimization support through directly rewriting the generated human-readable code

## Get Started
### Quick Start with Docker Image
For end users, simply use docker to compile your model and generate high-performance executable.
NNFusion supports and is tested on Ubuntu 16.04 and 18.04 with CUDA. You should install nvidia-docker on your device to do the following steps.
1. Pull docker image
`docker pull nnfusion/cuda:10.2-cudnn7-devel-ubuntu18.04`
2. Run docker container with the given image
```
docker run -t --name [YOUR_CONTAINER_NAME] -d nnfusion/cuda:10.2-cudnn7-devel-ubuntu18.04
docker start [YOUR_CONTAINER_NAME]
docker exec -it [YOUR_CONTAINER_NAME] bash
```
3. Put your model in the container
In host, you can use `docker cp host_path container_path` to copy your model into the container, or use `docker run -t -i -v <host_dir>:<container_dir>` to map the host dir to the container.
4. Compile Model
When model is prepared, we can compile model in the container.
```
cd root
nnfusion path/[YOUR_MODEL_FILE]
```
5. Run Compiled Model
```
cd root/nnfusion_rt/cuda_codegen
make. && make -j
./main_test
```

### Build from Source Code
Researchers or contributors who want to do more research on optimizing model compilation, you can build NNFusion from source code.
If you want to use docker to build from source code, please refer to [Build Guide for Docker Users](https://github.com/microsoft/nnfusion/wiki/Build-from-Source-Code-with-Docker-Image).
If you want to build from source code in your native system, please firstly read [System Requirements](https://github.com/microsoft/nnfusion/wiki/Before-Started) and check. Then please refer to [Build Guide for Native Users](https://github.com/microsoft/nnfusion/wiki/Build-Guide).

# Contributing

This project welcomes contributions and suggestions.  Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit https://cla.opensource.microsoft.com.

When you submit a pull request, a CLA bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., status check, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

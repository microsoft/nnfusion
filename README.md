**NNFusion** (Neural Network Fusion) is a flexible and efficient DNN compiler that can generate high-performance executables from a DNN model description (e.g., TensorFlow frozen models or ONNX format). NNFusion is designed to facilitate full-stack model optimization, framework-free code generation, and new accelerator devices support.

## Who should consider using NNFusion?
- Developers who want to deploy their pre-trained model as a framework-free source code with minimum library dependencies.
- Developers who want to speed up the execution performance of their pre-defined or pre-trained DNN model.
- Researchers who want to quickly try new compiler optimization ideas or customize optimizations on some specific models.

## Highlight features
- Provide ahead-of-time and source-to-source (model-to-code) compilation to reduce runtime overhead and remove library/framework dependencies.
- Provide a full-stack optimization mechanism, including:
  - Data-flow graph optimizations, e.g., CSE, compile-time constant folding, etc.
  - Model-specific kernel selection, kernel co-scheduling and kernel fusion
  - Static memory layout and placement optimizations
- Support popular DNN model formats including TensorFlow, ONNX, and TorchScript, and commonly used devices like CPU, CUDA GPUs, and ROCm GPUs.
- Support customized optimization in an easier and efficient way, e.g., directly replacing hand-crafted kernels on the generated human-readable code.

## Get Started
### Quick Start with Docker Image
For end users, simply use docker to compile your model and generate high-performance executable.

NNFusion supports and is tested on Ubuntu 16.04 and 18.04 with a CUDA GPU equipped. 

You should install nvidia-docker on your device to do the following steps.

We will use a simple TensorFlow LSTM inference model as an example. You can download a frozen version from our model zoo:

`wget https://nnfusion.blob.core.windows.net/models/tensorflow/frozen_lstm_l8s8h256_bs1.pb`

To use your own model to get started, please refer to [Supported Models](https://github.com/microsoft/nnfusion/wiki/1.3-Supported-Models) to see whether it is supported and freeze your model according to [Freeze Your Model](https://github.com/microsoft/nnfusion/wiki/1.1-Freeze-TensorFlow-Models).

1. Pull docker image
`docker pull nnfusion/cuda:10.2-cudnn7-devel-ubuntu18.04`

2. Run docker container with the given image

```
docker run -t --name [YOUR_CONTAINER_NAME] -d nnfusion/cuda:10.2-cudnn7-devel-ubuntu18.04
docker start [YOUR_CONTAINER_NAME]
docker exec -it [YOUR_CONTAINER_NAME] bash
```
3. Put your model in the container

In host, you can use `docker cp host_path [YOUR_CONTAINER_NAME]:container_path` to copy your model into the container, or use `docker run -t -i -v <host_dir>:<container_dir>` to map the host dir to the container.

4. Compile Model

When model is prepared, we can compile model in the container and run it to see the performance.
```
cd root
nnfusion path/[YOUR_MODEL_FILE]
```
Note: 
If you are using an ONNX model, the compile command will be  `nnfusion path/[YOUR_MODEL_FILE] -f onnx`

5. Build and Run Compiled Model

```
cd root/nnfusion_rt/cuda_codegen
cmake. && make -j
./main_test
```
6. The output of NNFusion should be Tensors with value and model iteration times.
For more detailed information on NNFusion usage, please refer to [NNFusion Usage](https://github.com/microsoft/nnfusion/wiki/3.-Compile-a-Tensorflow-model-with-NNFusion).

### Build from Source Code
Researchers or contributors who want to do more research on optimizing model compilation, you can build NNFusion from source code.
To build from source code, please read the following documents:
1. Read [Before Started](https://github.com/microsoft/nnfusion/wiki/1.-Before-Started) page to see supported CUDA GPUs and required libs. 
2. Read [Build Guide](https://github.com/microsoft/nnfusion/wiki/2.-Build-Guide) for more information on how to build and install NNFusion in your native system or in the docker container.
3. After building and installing NNFusion, please refer to [Compile Guide and Tool Usage](https://github.com/microsoft/nnfusion/wiki/3.-Compile-a-Tensorflow-model-with-NNFusion) to learn how to compile or optimize a DNN model.

### Performance

To learn how much performance improvement that NNFusion can achive on some typical DNN models, please refer to the [README page](https://github.com/microsoft/nnfusion/blob/osdi20_artifact/artifacts/README.md) at our OSDI'20 artifact branch. 

# Contributing

This project welcomes contributions and suggestions.  Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit https://cla.opensource.microsoft.com.

To contribute, please refer to [Contribution Guide](https://github.com/microsoft/nnfusion/wiki/4.-Guide-for-Contributors) to see more details.

When you submit a pull request, a CLA bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., status check, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

# Reference
Please cite NNFusion or Rammer in your publications if it helps your research:
```
@inproceedings {rammer-osdi20,
author = {Lingxiao Ma and Zhiqiang Xie and Zhi Yang and Jilong Xue and Youshan Miao and Wei Cui and Wenxiang Hu and Fan Yang and Lintao Zhang and Lidong Zhou},
title = {Rammer: Enabling Holistic Deep Learning Compiler Optimizations with rTasks},
booktitle = {14th {USENIX} Symposium on Operating Systems Design and Implementation ({OSDI} 20)},
year = {2020},
isbn = {978-1-939133-19-9},
pages = {881--897},
url = {https://www.usenix.org/conference/osdi20/presentation/ma},
publisher = {{USENIX} Association},
month = nov,
}
```

# Introduction
NNFusion is a flexible and efficient DNN compiler that can generate high-performance executables from a DNN model description. NNFusion is designed to facilitate DNN compiler research, with full stack of optimizations built-in and can target different accelerator devices.

### Major goal
- Framework independent: support ONNX, TensorFlow and PyTorch models through a source-to-source (model to code) compilation
- Innovation agility: provide a flexible and modular architecture to enable new compiler optimization research
- Hardware neutral: aims to be able to support existing and future 1st and 3rd party accelerator devices

### Highlight features
- Source-to-source compilation to remove framework overhead and dependency
- Full stack of optimization passes built-in
- Compile-time kernel co-scheduling abstraction for general accelerator devices
- Tight integration and co-optimization with communication, data loading and job scheduling
- Automatic kernel tuning through interacting with kernel tuning service
- Compatible with both hand-crafted kernel code and vendor-provided libraries
- Customized optimization support through directly rewriting the generated human-readable code

### Getting Started
To try NNFusion, please read the tutorial on [Compile a Tensorflow model with NNFusion](https://github.com/microsoft/nnfusion/wiki/Compile-a-Tensorflow-model-with-NNFusion).

### Build and Test
For more detailed building and testing information, please read the Build Guide.


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

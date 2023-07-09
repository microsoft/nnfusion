## Compile a Tensorflow model

The goal of this tutorial is to illustrate how to compile and optimize a DNN model, and demonstrate the performance improvement with NNFusion.

### Build & install NNFusion
We assume you already built and installed NNFusion in your environment with a CUDA GPU equipped, see supported CUDA GPUs from [Before Started](https://github.com/microsoft/nnfusion/blob/master/docs/Before-Started.md) page.
Please refer [Build Guide](https://github.com/microsoft/nnfusion/blob/master/docs/Build-Guide.md) for more information on how to build and install NNFusion.

###  Prepare model

We will use a simple TensorFlow LSTM inference model as an example ([models/tensorflow/lstm.py](https://github.com/microsoft/nnfusion/tree/master/models/tensorflow/lstm.py)), which has 8 layers of LSTM cells with sequence length of 8 and hidden size of 256.
Since the input of NNFusion is TensorFlow frozen model, we first need to use TensorFlow built-in freezing functionality to freeze this model into a protobuf file. The detailed instruction on how to freeze a model can refer [Freeze TensorFlow model](https://github.com/microsoft/nnfusion/blob/master/docs/Freeze-TensorFlow-Models.md).

Here in this tutorial, we can just download a frozen version from our model zoo:

```
wget https://nnfusion.blob.core.windows.net/models/tensorflow/frozen_lstm_l8s8h256_bs1.pb
```
### Compile model with NNFusion

Compile the LSTM frozen model with NNFusion (assume NNFusion binary is installed in your environment PATH):
```
nnfusion frozen_lstm_l8s8h256_bs1.pb
```
After the compilation, the end-to-end generated code is located under nnfusion_rt/cuda_codegen/ folder:
```
nnfusion_rt/cuda_codegen/
├── CMakeLists.txt
├── Constant
├── main_test.cpp
├── nnfusion_rt.cu
└── nnfusion_rt.h
```
The end-to-end code of this model is in file _nnfusion_rt.cu_, which is wrapped into a single C function named _kernel_entry_. 
There is a demo application generated in _main_test.cpp_, which is used to run and test performance.

## Run the compiled model
This *required* CUDA environment in native system.

To run this demo, NNFusion also generated a _CMakeLists.txt_ file to build the project. Note that, if your GPUs architecture is not in our default supporting list, please add the right "gencode" into CUDA_NVCC_FLAGS inside the CMakeLists.txt, eg: "_-gencode arch=compute_37,code=sm_37_" for K80.

Build and run the test:
```
cd nnfusion_rt/cuda_codegen && cmake . && make -j
./main_test
```
We set the value of input tensors to one. 

You can easily change the input tensor values by modifying `main_test.cpp` under the `cuda_codegen` folder.

Then NNFusion will run the model for 100 times and calculate the average latency. The output log containing both result tensor values and execution latencies will look like:
```
Result_2261_0: 
8.921492e-03 1.182089e-02 8.937407e-03 7.932202e-03 1.574193e-02 3.844392e-03 -1.505094e-02 -1.112035e-02 5.026605e-03 -8.032203e-03  .. (size = 256, ends with 1.357487e-02);
Iteration time 2.990464 ms
...
Iteration time 2.700096 ms
Iteration time 2.702432 ms
Summary: [min, max, mean] = [2.690368, 6.759712, 2.918306] ms
```

So far, we demonstrated how to compile and optimize a TensorFlow model with NNFusion. Note that, by default, NNFusion only applied a set of optimizations (e.g., kernel fusion). 

For some models, we can further optimize the performance by explicitly enabling some experimental optimization passes. For example, our tutorial on the BlockFusion optimization [OSDI RAMMER Tutorial](https://github.com/microsoft/nnfusion/blob/osdi20_artifact/artifacts/get_started_tutorial/README_GET_STARTED.md) shows how to optimize this model for a further 6x performance improvement. 

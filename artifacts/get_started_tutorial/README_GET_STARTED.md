# Get Started Tutorial: Compile a TensorFlow LSTM model with NNFusion (Rammer) 

We assume you already build and install NNFusion compiler folloing the *Environment Preparation* section in [README.md](../README.md).

The goal of this tutorial is to demonstrate how to compile and optimize a typical DNN model, and showcase the performance improvement with Rammer compiler.

## Freeze a TensorFlow model and run TensorFlow baseline

The input of NNFusion can be Tensorflow frozen model, ONNX model, TorchScript model. Here we use a TensorFlow LSTM inference model (*model/lstm_inference.py*) and freeze it into a protobuf file. 
As NNFusion currently only support TensorFlow 1.14 model format, we use TensorFlow built-in freezing tool to freeze the model:

```bash
# install tensorflow-gpu 1.14.0
pip install tensorflow-gpu==1.14.0
# freeze model
cd ~/nnfusion/artifacts/get_started_tutorial/model/
python lstm_inference.py --frozen_file frozen_lstm_l8s8h256_bs1.pb --num_iter 5

# run const folding to optimize the frozen graph (it also can be done in NNFusion's runtime const folding pass (-fconst_folding_backend=CUDA))
python tf_run_const_folding.py --file frozen_lstm_l8s8h256_bs1.pb

cd ../
```

After that, we can get the frozen graph pb file (frozen_lstm_l8s8h256_bs1.const_folded.pb), the ground truth result and the performance of TensorFlow.

```
mul_191:0
[ 0.00892149  0.01182088  0.0089374   0.0079322   0.01574193  0.00384439
 -0.01505094 -0.01112035  0.00502661 -0.0080322 ] ...(size= 256 end with 0.01357487 )
Iteration time 9.372234 ms
Iteration time 9.388685 ms
Iteration time 9.338617 ms
Iteration time 8.921146 ms
Iteration time 8.526325 ms
Summary: [min, max, mean] = [8.526325, 9.388685, 9.109402] ms
```

## Run RammerBase

As introduced in our paper, *RammerBase* referes to NNFusion baseline implementation without the Rammer optimizations. 
Note that, RammerBase already includes the common graph optimizations (such as CSE, constant folding, element-wise kernel fusion, etc) in the state-of-the-art compilers and frameworks.
We use RammerBase to show how much performance can be improved with these optimizations.

```bash
# compile the frozen model and generate the exectuable source code
nnfusion model/frozen_lstm_l8s8h256_bs1.const_folded.pb -f tensorflow -fkernel_fusion_level=3 -fconst_folding_backend=CUDA -fproduct_name="Tesla V100-PCIE-16GB" -fwarmup_step=5 -frun_step=5

# after the compilation, the end-to-end code is generated under the nnfusion_rt/ folder
mv nnfusion_rt lstm_rammerbase_cudalib
cd lstm_rammerbase_cudalib/cuda_codegen/

# build model code and run
cmake . && make -j
./main_test
cd ../..
```

It will output the following logs:

```
Result_2110_0:
8.921492e-03 1.182089e-02 8.937407e-03 7.932202e-03 1.574193e-02 3.844392e-03 -1.505094e-02 -1.112035e-02 5.026605e-03 -8.032203e-03  .. (size = 256, ends with 1.357487e-02);
Iteration time 2.530048 ms
Iteration time 2.535136 ms
Iteration time 2.542944 ms
Iteration time 2.527776 ms
Iteration time 2.529440 ms
Summary: [min, max, mean] = [2.527776, 2.542944, 2.554451] ms
```

We achieves 3.5x speedup over TensorFlow by a series of common compiler optimizations (e.g., kernel fusion) and moving dataflow graph scheduling from runtime to compile time.

## Run Rammer with Library Kernels

We now evaluate Rammer through running the same experiment in the above except turning on the Rammer optimization through sepcifing **-fblockfusion_level=1**:

```bash
nnfusion model/frozen_lstm_l8s8h256_bs1.const_folded.pb -f tensorflow -fkernel_fusion_level=3 -fblockfusion_level=1 -fconst_folding_backend=CUDA -fproduct_name="Tesla V100-PCIE-16GB" -fwarmup_step=5 -frun_step=5

# build model code and run
mv nnfusion_rt lstm_rammer_cudalib
cd lstm_rammer_cudalib/cuda_codegen/
cmake . && make -j
./main_test

cd ../..
```
It will output the following logs:

```
Result_2110_0:
8.921488e-03 1.182088e-02 8.937404e-03 7.932203e-03 1.574193e-02 3.844394e-03 -1.505095e-02 -1.112035e-02 5.026603e-03 -8.032203e-03  .. (size = 256, ends with 1.357487e-02);
Iteration time 2.353216 ms
Iteration time 2.351424 ms
Iteration time 2.352992 ms
Iteration time 2.370496 ms
Iteration time 2.352160 ms
Summary: [min, max, mean] = [2.351424, 2.370496, 2.377466] ms
```

As it shows, we can achieve a little bit better performance, but only in a small margin. This is because that most current operators like Dot (Matrix Multiplication) are implemented as calling CUDA libraries like cuBLAS and cuDNN, which are close-sourced kernels and cannot be converted to rOperator.

In the next section, we will demonstrate if we inject rOperator compatiable kernels into NNFusion, how much performance gain can be achived by Rammer through the holistic optimization.

## Run Rammer with rOperator kernels

The rOperator needs source code of kernels, thus we mannuly implemented kernel code for each operator used in LSTM model and then we convert it to rOperator kernels.
These generated kernels will be injected into a kernel database, where NNFusion can directly load. 
Below is a script to generate and inject kernels for NNFusion:

### Prepare kernel Database

```bash
cd kernel_db/codegen_scripts/
python manual_dense_codegen.py

cd ../kernel_db_scripts
bash init.sh
python convert_tvm.py ../example_kernel_db.json
cd ../../
```

After that, you can get a kernel database file in ```~/.cache/nnfusion/kernel_cache.db```. NNFusion will automatically detect this path and import these kernels.

### Model Run: RammerBase with rKernels
Given the new kernels generated for Rammer, as a baseline, we still first run the RammerBase with these new kernels through sepcifing **-fblockfusion_level=0**:

```bash
nnfusion model/frozen_lstm_l8s8h256_bs1.const_folded.pb -f tensorflow -fkernel_fusion_level=3 -fblockfusion_level=0 -fconst_folding_backend=CUDA -fproduct_name="Tesla V100-PCIE-16GB" -fwarmup_step=5 -frun_step=5

# build model code and run
mv nnfusion_rt lstm_rammerbase && cd lstm_rammerbase/cuda_codegen/
cmake . && make -j
./main_test

cd ../..
```
It will output the following logs:
```
Result_2110_0:
8.921486e-03 1.182089e-02 8.937402e-03 7.932202e-03 1.574193e-02 3.844391e-03 -1.505094e-02 -1.112035e-02 5.026604e-03 -8.032196e-03  .. (size = 256, ends with 1.357487e-02);
Iteration time 3.457408 ms
Iteration time 3.464288 ms
Iteration time 3.458144 ms
Iteration time 3.468736 ms
Iteration time 3.453728 ms
Summary: [min, max, mean] = [3.453728, 3.468736, 3.482899] ms
```
As we can see, the overall performance is slower than RammerBase with cuBLAS kernels, this is because the manually implemented kernel is not as good as cuBLAS. 
However, even the individual kernel is not optimal, the following experiment will demonstrate the end-to-end performance gain achived by Rammer's complication.

### Model Run: Rammer with rKernels

We now re-evaluate Rammer through running the same experiment in the above with the Rammer optimization through sepcifing **-fblockfusion_level=1**:

```bash
nnfusion model/frozen_lstm_l8s8h256_bs1.const_folded.pb -f tensorflow -fkernel_fusion_level=3 -fblockfusion_level=1 -fconst_folding_backend=CUDA -fproduct_name="Tesla V100-PCIE-16GB" -fwarmup_step=5 -frun_step=5

# build model code and run
mv nnfusion_rt lstm_rammer && cd lstm_rammer/cuda_codegen/
cmake . && make -j
./main_test

cd ../..
```

It will output the following logs:

```
Result_2110_0:
8.921487e-03 1.182089e-02 8.937405e-03 7.932202e-03 1.574193e-02 3.844393e-03 -1.505094e-02 -1.112035e-02 5.026609e-03 -8.032201e-03  .. (size = 256, ends with 1.357487e-02);
Iteration time 0.338592 ms
Iteration time 0.331904 ms
Iteration time 0.336160 ms
Iteration time 0.333568 ms
Iteration time 0.343648 ms
Summary: [min, max, mean] = [0.331904, 0.343648, 0.357389] ms
```

We further achieve 6.6x speedup against Rammer-CUDALib by enabling holistic optimization on the whole model, and reduce the end-to-end inference latency from 9.109402 ms in TensorFlow to 0.357389 ms.
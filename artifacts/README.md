# OSDI'23 Artifacts Evaluation

## 0. Overview
  This code branch is used for OSDI'23 Artifact Evaluation of paper #847, titled "Welder: Scheduling Deep Learning Memory Access via Tile-graph".

- Artifacts Available:
    - Most Welder related code are open-sourced under this repo and. Some Welder related code are implemented in the welder branch of the TVM and NNFusion repo.

- Artifacts Functional:
    - *Documentation*: the following of documents include detailed guidelines on how to build, install, test Welder and the experiments to compare with other baselines.
    - *Completeness*: the source code under "welder/" folder includes all the key components of Welder described in the paper.
    - *Exercisability*: under the *artifacts* folder, we prepare all the script and data to reproduce the experiments in individual folders named by the figure name in paper.

- Results Reproduced:
    - To reproduce the main results presented in our paper, we provide a Docker image containing the environments. As the GraphCore, ROCm environments are internal resources with restrict accessibility, we use the CUDA GPUs (NVIDIA Tesla V100 GPU) environment to reproduce the main results. We provide detailed guidelines to help reproduce the results step by step. For the rest inaccessible environments, We also provide detailed guideline to help reproduce the results step by step.

## 1. Evaluation Setup
To ease the process of installing all the dependencies, baseline softwares, and Welder code, we provide a Dockerfile and a simple guideline to build a Docker image with all of above installed.

```bash
cd welder
# build the image
sudo docker build -t welder_cuda .
# run the container
sudo nvidia-docker run -it --cap-add=SYS_ADMIN --network=host --name welder_test welder_cuda bash
```

## 2. Paper result reproduce

We have provided tuning logs and pre-compiled models under folder /sharepoint in the Docker so that reviewers can quickly reproduce results in the paper. Here is a list we provide:

| Name      | Description                                              | Commands                      |
| --------- | -------------------------------------------------------- | ----------------------------- |
| Figure1   | onnxruntime memory performance for different models     | [Figure1](#f1) |
| Figure9   | Model inference performance on V100 FP32                 | [Figure9_10](#f2)                    |
| Figure10a | Model inference performance on V100 FP16 (TensorCore)     | [Figure9_10](#f2)                    |
| Figure10b | Model inference performance on V100 FP16 (No TensorCore) | [Figure10b](#f3)                     |
| Figure11  | Latency, kernel count, global memory transaction and IRS | [Figure11](#f4)                      |
| Table3    | Performance for WELDER and FasterTransformer             | [Table3](#f5)                        |
| Table4    | Compilation time of Ansor and Welder                     | [Table4](#f6)                        |
| Table5    | Performance on compute intensive models                  | [Table5](#f7)                        |
| Table6    | Scale-up large DNN models to host memory (GPU)           | [Table6](#f8)                        |

Note that results in Figure12/part of Table 6 requires ROCM-GPU/GraphCore IPU environments which is not directly available here.

### <a id="f1">Figure1 </a>

The run instruction is

```bash
python run_all.py
```

### <a id="f2">Figure9-10</a>

The run instructions for Welder, onnxruntime, pytorch, tensorrt and Rammer are

```bash
python profile_rammer_all.py
python profile_ort_all.py
python profile_torch_all.py
python profile_welder_all.py
python profile_trt_all.py
```

The run instruction for Ansor is, but requires additional action before running it.

```bash
# Our tune log for Ansor only applies for this version.
cd /root/tvm/build && git checkout v0.9.0 && make -j
# after switching branch
cd -
python profile_ansor_all.py
# don't forget to get back
cd /root/tvm/build && git checkout welder && make -j
```

### <a id="f3">Figure10b</a>

The run instruction is

```bash
python profile_welder_no_tc.py
```

### <a id="f4"> Figure 11</a>

The run instructions are

```bash
# measure latency, IRS and kernel count
python get_IRS.py
# measure memory perf
python get_metrics.py
```

### <a id="f5">Table3</a>

The run instruction is

```bash
python run_ft_cpp_all.py
```

### <a id="f6">Table4</a>

The run instruction is

```bash
python estimate_run_time_welder.py
python estimate_run_time_ansor.py
```

### <a id="f7">Table5</a>

The run instruction is

```bash
python run_all.py
```

### <a id="f8">Table6</a>

The run instruction is

```bash
bash run_all.sh
```

## 3. Getting started with Welder

Despite using the logs provided above, you can also run welder from scratch. To compile a model with Welder, there are several steps.

### Step1: Create a ONNX model file:

```bash
python torch2onnx.py MODEL --prefix PREFIX [--bs BATCHSIZE] [--fp16]
```

To generate an ONNX model, we first use the script torch2onnx.py to generate an onnx file under the PREFIX folder.

The MODEL parameter can be one of the ten models used in the paper (bert, vit, swin_transformer, BSRN, NAFNet, Restormer, mobilevit, Conformer, mobilenet and NeRF).

Default batchsize is 1, it can be set with --bs flag. The default datatype is float32, if --fp16 is used, the datatype will be float16.

After running this command, The PREFIX folder will be created which contains a model.onnx file. This PREFIX will be used in the following Welder's compilation steps. Some other baselines will also use this PREFIX as the workspace.

### Step2: Compile ONNX with Welder

Afther the PREFIX folder is created, goto this folder and run the following command

```bash
nnfusion model.onnx -f onnx -ftune_output_file=model.json &&
python3 -m run_compiler model.json tuned.json --topk 20 --arch V100 &&
nnfusion model.onnx -f onnx -ftune_output_file=/dev/null -ftune_input_file=tuned.json &&
rm -rf nnfusion_rt/cuda_codegen/build/ &&
cmake -S nnfusion_rt/cuda_codegen/ -B nnfusion_rt/cuda_codegen/build/ &&
make -C nnfusion_rt/cuda_codegen/build/
```

This command looks complicated, as it includes some compile flags. We will briefly explain what does each line do. The first line will convert model.onnx into model.json which is the compute graph definition. The second line converts model.json into tuned.json which contains tuned kernels, --topk 20 and --arch V100 indicates that 20 trails is made for each task(subgraph) and V100 GPU is the target. The third line organized kernels in tuned.json constructs CUDA/C code. The last three line use device compilers(nvcc) to compile the CUDA/C code into binary.

Specially, when reproducing results in the paper, special flags will be added. The 3 included cases are: bert, fp32, bs=1,64 and swin_transformer, fp16, bs=1. In this three cases, we add "--fusion_skiplist=Dot" after the nnfusion command in both line 1 and line 3:

```bash
nnfusion model.onnx -f onnx -ftune_output_file=model.json -ffusion_skiplist=Dot &&
python3 -m run_compiler model.json tuned.json --topk 20 --arch V100 &&
nnfusion model.onnx -f onnx -ftune_output_file=/dev/null -ftune_input_file=tuned.json -ffusion_skiplist=Dot &&
rm -rf nnfusion_rt/cuda_codegen/build/ &&
cmake -S nnfusion_rt/cuda_codegen/ -B nnfusion_rt/cuda_codegen/build/ &&
make -C nnfusion_rt/cuda_codegen/build/
```


Ths flag will lower some Dot kernels to CUDA library (cublas) which performs better than generated kernels in these 3 cases.

### Step3: Evaluate Latency and correctness.

After running the previous command, you can profile the latency of the welder's generated model:

```bash
# to evaluate inference performance, you can directly use an executable
cd PREFIX/nnfusion_rt/cuda_codegen
./build/main_test

# OR use python script which feeds data with pytorch and ctypes
python3 run_welder.py PREFIX
```

To check the correctness of Welder's compiled model, you can run the following command to compare Welder's output with onnx-runtime's output.

```bash
python3 test_acc.py PREFIX
```

You can also run other baselines on this model:

```bash
# torch
python run_torch.py MODEL [--bs BATCHSIZE] [--fp16]
# TensorRT
python run_trt.py --prefix PREFIX [--fp16]
# onnxruntime
python run_onnxrt.py --prefix PREFIX
# Ansor, note that Ansor requires about one day to tune for a model
python run_ansor.py --prefix PREFIX
# Astitch
python3 run_blade.py MODEL [--bs BATCHSIZE] [--fp16]
```

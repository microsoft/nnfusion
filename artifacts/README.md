# OSDI'23 Cocktailer Artifacts Evaluation

## 0. Overview
This code branch is used for OSDI'23 Artifact Evaluation of paper #628, titled "Cocktailer: Analyzing and Optimizing Dynamic Control Flow in Deep Learning".

### Evaluation Setup
* Artifacts Available:
    * All Cocktailer related code are available under NNFusion open-source project located in: [https://github.com/microsoft/nnfusion/tree/cocktailer_artifact](https://github.com/microsoft/nnfusion/tree/cocktailer_artifact)
* Artifacts Functional:
    * *Documentation*: the following of documents include detailed guidelines on how to build, install, test Cocktailer and the experiments to compare with other baselines.
    * *Completeness*: the [C++ part](..) of Cocktailer has been merged into NNFusion in this branch, and the [Python part](ast_analyzer) is available in this artifact.
    * *Exercisability*: under the *artifacts* folder, we prepare all the script and data to reproduce the experiements in individual folders named by the figure name in paper.
* Results Reproduced:
    * To reproduce the main results presented in our paper, we provide Docker images containing all the environments and baseline software, and machines with the same configurations as we used in paper evaluation. We also provide detailed guideline to help reproduce the results step by step. 

## 1. Environment Preparation
### NVIDIA GPU
Please follow the instructions in [INSTALL.md](INSTALL.md) or use the following docker-based script to build and install Cocktailer.
```bash
cd $YOUR_DIR_FOR_NNFUSION
git clone https://github.com/microsoft/nnfusion.git --branch cocktailer_artifact --single-branch
cd nnfusion/artifacts
docker build -t cocktailer -f env/Dockerfile.nv .
chmod 777 $YOUR_DIR_FOR_NNFUSION/nnfusion
docker run -it --gpus all --name cocktailer-ae -v $YOUR_DIR_FOR_NNFUSION/nnfusion:/root/nnfusion --shm-size="32g" -w /root/nnfusion/artifacts cocktailer:latest /bin/bash
# run inside docker
bash ./env/install_in_docker.sh
```

### AMD GPU
Please prepare four dockers for running JAX, TensorFlow, TVM, PyTorch \& Cocktailer respectively.
* download code
    ```bash
    cd $YOUR_DIR_FOR_NNFUSION
    git clone https://github.com/microsoft/nnfusion.git --branch cocktailer_artifact --single-branch
    ```
* Build and run jax docker (the result image is `jax-rocm:latest`)
    ```bash
    cd $YOUR_DIR_FOR_NNFUSION/nnfusion/artifacts
    mkdir third-party && cd third-party
    git clone https://github.com/google/jax.git
    cd jax
    git checkout 0282b4bfad
    git apply ../../env/jax.rocm.patch
    ./build/rocm/ci_build.sh --keep_image bash -c "./build/rocm/build_rocm.sh"
    docker run -it --device=/dev/kfd --device=/dev/dri --name jax-ae -v $YOUR_DIR_FOR_NNFUSION/nnfusion:/root/nnfusion -w /root/nnfusion/artifacts -e ARTIFACT_ROOT=/root/nnfusion/artifacts jax-rocm:latest /bin/bash
    ```
* Pull and run TensorFlow docker
    ```bash
    docker pull rocm/tensorflow:rocm4.3.1-tf1.15-dev
    docker run -it --device=/dev/kfd --device=/dev/dri --name tf-ae -v $YOUR_DIR_FOR_NNFUSION/nnfusion:/root/nnfusion -w /root/nnfusion/artifacts -e ARTIFACT_ROOT=/root/nnfusion/artifacts rocm/tensorflow:rocm4.3.1-tf1.15-dev /bin/bash
    ```
* Build and run TVM docker
    ```bash
    mkdir $YOUR_DIR_FOR_NNFUSION/kernel_db
    cd $YOUR_DIR_FOR_NNFUSION/nnfusion/artifacts
    docker build -t tvm_rocm_cuda:latest -f env/Dockerfile.tvm.rocm .
    docker run -it --device=/dev/kfd --device=/dev/dri --name tvm-ae -v $YOUR_DIR_FOR_NNFUSION/kernel_db:/root/.cache/nnfusion -v $YOUR_DIR_FOR_NNFUSION/nnfusion:/root/nnfusion -w /root/nnfusion/artifacts -e ARTIFACT_ROOT=/root/nnfusion/artifacts tvm_rocm_cuda /bin/bash
    ```
* Build and run cocktailer docker
    ```bash
    cd $YOUR_DIR_FOR_NNFUSION/nnfusion/artifacts
    docker build -t cocktailer:latest -f env/Dockerfile.rocm .
    docker run -it --device=/dev/kfd --device=/dev/dri --name cocktailer-ae -v $YOUR_DIR_FOR_NNFUSION/kernel_db:/root/.cache/nnfusion -v $YOUR_DIR_FOR_NNFUSION/nnfusion:/root/nnfusion -w /root/nnfusion/artifacts -e ARTIFACT_ROOT=/root/nnfusion/artifacts cocktailer /bin/bash
    # run inside docker
    bash ./env/install_in_rocm_docker.sh
    ```

## 2. Getting Started with a Simple Example

* Go to the *get_started_tutorial/* folder and follow [README_GET_STARTED.md](get_started_tutorial/README_GET_STARTED.md).


## 3. Data and Kernel Preparation
* Download the input data and model weights from [zenodo](https://doi.org/10.5281/zenodo.7856472), unzip them and put them under the nnfusion/artifacts directory. The tree structure should be like:
    ```
    nnfusion
    ├── artifacts
    │   ├── data
    │   │   ├── attention
    │   │   ├── blockdrop
    │   │   ├── lstm
    │   │   ├── seq2seq
    │   │   ├── skipnet
    │   │   ├── sst
    │   │   └── tatoeba-eng-fra
    ```

* Generates all kernels for Cocktailer. More details can be found in [README_KERNEL_DB.md](kernel_db/README_KERNEL_DB.md).
    **NOTE**: this process will take about 20 minutes for each architecture if using the tuning result in the artifact, or longer if you want to re-tune the kernels.
    * NVIDIA GPU
        ```bash
        # assume running at nnfusion/artifacts directory
        cd kernel_db
        ./reproduce_kernel_db.sh
        mkdir -p /tmp/`whoami` && cp -r ~/.cache/nnfusion/* /tmp/`whoami`/
        ```
    * AMD GPU
        ```bash
        # assume running at nnfusion/artifacts directory of tvm docker
        cd kernel_db
        ./reproduce_rocm_kernel_db.sh
        ```

## 4. Reproducing Individual Experiement Results
**NOTE**: we provide a script named "run_nv_gpu.sh" to run the experiments except Figure20. You can go to `nnfusion/artifacts` directory and use `./run_nv_gpu.sh` to run the experiments. For Figure20, please follow the README.md in the `Figure20` directory.

| Experiments   | Figure # in Paper |  Script Location |
| -----------     | -----------  |  ----------- |
| #1. Control flow overhead in JAX | Figure 2 | N/A (use the results in Figure 15, 16, and 18) |
| #2. End-to-end DNN inference on NVIDIA V100 GPU | Figure 14 | [run.sh](Figure14/run.sh) |
| #3. Control flow overhead of models with loops | Figure 15 | [run.sh](Figure15/run.sh) |
| #4. Control flow overhead of models with branches | Figure 16 | [run.sh](Figure16/run.sh) |
| #5. Different ratio of executed layers | Figure 17 | [run.sh](Figure17/run.sh) |
| #6. Control flow overhead of RAE with recursion | Figure 18 | [run.sh](Figure18/run.sh) |
| #7. Breakdown of models with BS=1 | Figure 19 | [run.sh](Figure19/run.sh)|
| #8. End-to-end DNN inference on AMD MI100 GPU with BS=1 | Figure 20 | [README.md](Figure20/README.md) |

## 5. Reproduce the Figures in the paper
Copy the ROCM results to the NVIDIA GPU node and draw figures on the NVIDIA GPU node

```bash
cd $YOUR_DIR_FOR_NNFUSION/nnfusion/artifacts
# copy the logs in $YOUR_DIR_FOR_NNFUSION/nnfusion/artifacts/reproduce_results/Figure20 of the AMD GPU node to $YOUR_DIR_FOR_NNFUSION/nnfusion/artifacts/reproduce_results of the NVIDIA GPU node
cd plot && ./plot_all.sh && cd - 
```
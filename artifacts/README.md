# OSDI'23 Grinder Artifacts Evaluation

## 0. Overview
This code branch is used for OSDI'23 Artifact Evaluation of paper #628, titled "Grinder: Analysis and Optimization for Dynamic Control Flow in Deep Learning".

### Evaluation Setup
* Artifacts Available:
    * All Grinder related code are available under NNFusion open-source project located in: [https://github.com/microsoft/nnfusion/tree/TODO](https://github.com/microsoft/nnfusion/tree/TODO)
* Artifacts Functional:
    * *Documentation*: the following of documents include detailed guidelines on how to build, install, test Grinder and the experiments to compare with other baselines.
    * *Completeness*: the [C++ part](..) of Grinder has been merged into NNFusion in this branch, and the [Python part](ast_analyzer) is available in this artifact.
    * *Exercisability*: under the *artifacts* folder, we prepare all the script and data to reproduce the experiements in individual folders named by the figure name in paper.
* Results Reproduced:
    * To reproduce the main results presented in our paper, we provide Docker images containing all the environments and baseline software, and machines with the same configurations as we used in paper evaluation. We also provide detailed guideline to help reproduce the results step by step. 

## 1. Environment Preparation

**For AE Reviewers**:
1. The nico cluster we provide for artifact evaluation is managed by slurm. To run GPU-related commands, please use `srun --pty --exclusive` before the original command, which will submit the job to the compute node (nico[3-4]). For your convenience, we have included this prefix in our artifact but will remove it in the final version. If you are running the artifact on your own machine, please remember to remove the prefix.
2. Due to security concerns, we cannot provide the docker permission to reviewers. Instead, for NVIDIA GPU, we provide an account with all the dependencies installed, and for AMD GPU, we provide ssh access into the dockers. You can skip this environment preparation section.

## NVIDIA GPU
```bash
cd $YOUR_DIR_FOR_NNFUSION
git clone https://github.com/microsoft/nnfusion.git --branch TODO --single-branch
cd nnfusion/artifacts
docker build -t grinder -f env/Dockerfile.nv .
chmod 777 $YOUR_DIR_FOR_NNFUSION/nnfusion
docker run -it --gpus all --name grinder-ae -v $YOUR_DIR_FOR_NNFUSION/nnfusion:/root/nnfusion --shm-size="32g" -w /root/nnfusion/artifacts grinder:latest /bin/bash
# run inside docker
bash ./env/install_in_docker.sh
```

adapted (TODO: remove)
```bash
docker build --network=host -t grinder -f env/Dockerfile.nv .
docker run -it --gpus all --name heheda-grinder-ae -v /home/heheda/control_flow/nnfusion-docker:/root/nnfusion -v /home/heheda/control_flow/kernel_db.docker:/root/.cache/nnfusion -w /root/nnfusion/artifacts --privileged=true --shm-size="32g" --network=host grinder:latest /bin/bash
srun -w nico3 --pty --exclusive docker exec -it heheda-grinder-ae bash ./run_nv_gpu.sh
permission: chmod 777 the two folders, config not to /dev/shm
```

## AMD GPU
* download code and data
TODO
* Build jax docker
    ```bash
    cd $YOUR_DIR_FOR_NNFUSION/nnfusion/artifacts
    mkdir third-party && cd third-party
    git clone https://github.com/google/jax.git
    cd jax
    git checkout 0282b4bfad
    git apply ../../env/jax.rocm.patch
    ./build/rocm/ci_build.sh --keep_image bash -c "./build/rocm/build_rocm.sh"
    ```
* Build grinder docker
    ```bash
    cd $YOUR_DIR_FOR_NNFUSION/nnfusion/artifacts
    docker build -t grinder:latest -f env/Dockerfile.rocm --network=host .

    # run inside docker
    bash ./env/install_in_docker.sh
    ```
* Pull the TensorFlow docker
    ```bash
    docker pull rocm/tensorflow:rocm4.3.1-tf1.15-dev
    ```
* Build TVM docker
    ```bash
    cd $YOUR_DIR_FOR_NNFUSION/nnfusion/artifacts
    docker build -t tvm_rocm_cuda:latest -f env/Dockerfile.tvm.rocm --network=host .
    ```

## 2. Getting Started with a Simple Example

* Go to the *get_started_tutorial/* folder and follow [README_GET_STARTED.md](get_started_tutorial/README_GET_STARTED.md).


## 3. Data and Kernel Preparation
* Download the input data and model weights from TODO, unzip them and put them under the nnfusion/artifacts directory. The tree structure should be like:
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
* Generates all kernels for Grinder. More details can be found in [README_KERNEL_DB.md](kernel_db/README_KERNEL_DB.md).
    
    **NOTE**: this process will take about 20 minutes if using the tuning result in the artifact, or much longer if you want to re-tune the kernels.
    ```bash
    # assume running at nnfusion/artifacts directory

    # On Nvidia GPU node
    cd kernel_db
    srun --pty --exclusive ./reproduce_kernel_db.sh
    srun -w nico3 --pty bash -c "mkdir -p /tmp/`whoami` && rsync -avz nico0:~/.cache/nnfusion/* /tmp/`whoami`/"
    srun -w nico4 --pty bash -c "mkdir -p /tmp/`whoami` && rsync -avz nico0:~/.cache/nnfusion/* /tmp/`whoami`/"
    # On AMD GPU node
    TODO
    ```

## 4. Reproducing Individual Experiement Results
**NOTE**: we provide a script named "run_nv_gpu.sh" to run the experiments except Figure19. You can go to `nnfusion/artifacts` directory and use `./run_nv_gpu.sh` to run the experiments. TODO: explain the run of Figure 19.

**For AE Reviewers**: Please use `srun --pty --exclusive ./run_nv_gpu.sh ` to submit the jobs to the compute node of the provided cluster.

| Experiments   | Figure # in Paper |  Script Location |
| -----------     | -----------  |  ----------- |
| #1. Control flow overhead in JAX | Figure 2 | N/A (use the results in Figure 15, 16, and 18) |
| #2. End-to-end DNN inference on NVIDIA V100 GPU | Figure 14 | [run.sh](Figure14/run.sh) |
| #3. Control flow overhead of models with loops | Figure 15 | [run.sh](Figure15/run.sh) |
| #4. Control flow overhead of models with branches | Figure 16 | [run.sh](Figure16/run.sh) |
| #5. Different ratio of executed layers | Figure 17 | [run.sh](Figure17/run.sh) |
| #6. Control flow overhead of RAE with recursion | Figure 18 | [run.sh](Figure18/run.sh) |
| #7. End-to-end DNN inference on ROCm MI100 GPU with BS=1 | Figure 19 | [run.sh](Figure19/run.sh) TODO |
| #8. Breakdown of models with BS=1 | Figure 20 | [run.sh](Figure20/run.sh)|

## 5. Reproduce the Figures in the paper
TODO: collect result of Figure 19 with SCP

```bash
 cd plot && ./plot_nv.sh && cd - 
```
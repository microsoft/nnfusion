# OSDI'22 Artifacts Evaluation

## 0. Overview
  This code branch is used for OSDI'22 Artifact Evaluation of paper #158, titled "Roller: Fast and Efficient Tensor Compilation for Deep 
              Learning". 


### Evaluation Setup

- Artifacts Available: 
    - All Roller related code are available under NNFusion open-source project located in: https://github.com/microsoft/nnfusion/tree/osdi22_artifact/artifacts/roller

- Artifacts Functional:
    - *Documentation*: the following of documents include detailed guidelines on how to build, install, test Roller and the experiments to compare with other baselines.
    - *Completeness*: the source code under "roller/" folder includes all the key components of Roller described in the paper.
    - *Exercisability*: under the *artifacts* folder, we prepare all the script and data to reproduce the experiements in individual folders named by the figure name in paper.

- Results Reproduced:
    - To reproduce the main results presented in our paper, we provide a Docker image containing all the environments and baseline software, and an Azure NC24s_v3 VM with the same configurations as we used in paper evaluation. As the GraphCore, ROCm and NVIDIA K80 environments are internal resoruces with resitrict accessbility, we use the CUDA GPUs (NVIDIA Tesla V100 GPU) environment to reproduce the main results. We provide detailed guidelines to help reproduce the results step by step. For the rest inaccessible environments, we also provide the source code and recent running traces that one can use to reproduce on their own environment if possible. 


## 1. Environment Preparation

To ease the process of installing all the dependencies, baseline softwares, and Roller code, we provide a Dockerfile and a simple guideline to build a Docker image with all of above installed:

* First, checkout the source code:
    ```
    git clone https://github.com/microsoft/nnfusion nnfusion -b osdi22_artifact
    ```
* Build a docker container named as *roller_cuda*:
    ```
    cd nnfusion/artifacts/
    bash scripts/build_container.sh
    ```
* Run into the container and execute a bash environment. If you want to reset the container, you can execute the *remove_container.sh* before this:
    ```
    bash scripts/open_container_old_docker.sh
    ```
* Now, supposedly you are inside the container and located in */root/nnfusion/*. Then you can insatll all baseline software (e.g., TensorFlow 1.15.2, Tensor RT-7.0.0, TVM-0.7) through:
    ```
    bash artifacts/scripts/build_and_install_deps.sh
    ```
* For better evaluation figure produced, the font named Times-New-Roman need to be installed by this command:
    ```
    apt install ttf-mscorefonts-installer
    ```

Now, you are ready to go!

## 2. Getting Started with a Simple Example

 - Go to the *get_started_tutorial/* folder and follow [README_GET_STARTED.md](get_started_tutorial/README_GET_STARTED.md).


## 3. Reproducing Individual Experiement Results

* Note: we provide a script named "run_all.sh" to run each experiment under each folder. The detail guidelines and explainations for each step are also commented in the script.

| Experiments   | Figure # in Paper |  Script Location | Instructions | 
| -----------     | -----------  |  ----------- |    ----------- |              
| #1. Operator performance  | Figure 10 | nnfusion/artifacts/figure10 | [run_all.sh](figure10/run_all.sh)  |
| #2. Compilation time for each operator | Figure 11 | nnfusion/artifacts/figure11 | [run_all.sh](figure11/run_all.sh) |
| #3. Scale-out MatMul operator | Figure 12 | nnfusion/artifacts/figure12 | [run_all.sh](figure12/run_all.sh) |
| #4. Scale-out Conv2d operator | Figure 13 | nnfusion/artifacts/figure13 | [run_all.sh](figure13/run_all.sh) |
| #5. Compilation time for MatMul and Conv2d | Figure 14 | nnfusion/artifacts/figure14 | [run_all.sh](figure14/run_all.sh) |
| #6. Compile on TensorCore | Figure 15 | nnfusion/artifacts/figure15 | [run_all.sh](figure15/run_all.sh) |
| #7. Performance for small operators | Figure 16 | nnfusion/artifacts/figure16 | [run_all.sh](figure16/run_all.sh) |
| #8. Performance for operators with irregular shapes | Figure 17 | nnfusion/artifacts/figure17 | [run_all.sh](figure17/run_all.sh) |
| #9. End-to-end model performance | Table2 | nnfusion/artifacts/table2 | [run_all.sh](table2/run_all.sh) |
| #10. Operator performance on NVIDIA K80 GPUs | Table3 | nnfusion/artifacts/table3 | [run_all.sh](table3/run_all.sh) |
| #11. Operator performance on ROCm MI50 GPUs | Table4 | nnfusion/artifacts/table4 | [run_all.sh](table4/run_all.sh) |
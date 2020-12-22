This guide is tested under Ubuntu 16.04 & 18.04, with or without Nvidia/AMD GPU installed.

## Build from Source Code:
   
   Users can build from source code in your native system or in a docker container.
1. Clone a copy of the repo:

    `git clone https://github.com/microsoft/nnfusion.git`


2. Install dependencies:

    Install dependencies by use the script (need sudo):

    `./maint/script/install_dependency.sh`

3. Use cmake to config and build:

    `mkdir build && cd build && cmake .. && make -j6`
4. Use nnfusion CLI:

    Nnfusion CLI could be installed in system by make install, or use `./build/src/tools/nnfusion/nnfusion` instead. You can refer to [NNFusion Client Tutorial](https://github.com/microsoft/nnfusion/blob/master/docs/NNFusion-CLI-Interface.md) to learn about NNFusion client.

5. Run test:

    Please see [End to End Test](https://github.com/microsoft/nnfusion/blob/master/docs/End-to-end-Test.md).

After building and installing NNFusion, please refer to [Compile Guide and Tool Usage](https://github.com/microsoft/nnfusion/blob/master/docs/Compile-a-Tensorflow-model-with-NNFusion.md) to learn how to compile or optimize a DNN model.





This tutorial demonstrates how to use NNFusion to compile a TensorFlow model and tune each operator in this model to generate the end-to-end source code.

1. Pull NNFusion docker image with kernel tuning tools (e.g., Antares) enabled: 
    ``` 
    docker pull nnfusion/cuda:10.2-cudnn7-devel-ubuntu18.04-antares`
    ```
2. Run docker container with the given image
    ```
    docker run --runtime=nvidia -t --name [YOUR_CONTAINER_NAME] -d nnfusion/cuda:10.2-cudnn7-devel-ubuntu18.04-antares
    docker start [YOUR_CONTAINER_NAME]
    docker exec -it [YOUR_CONTAINER_NAME] bash
    ```
3. Inside the docker container, we will use a simple TensorFlow LSTM inference model as an example. You can download a frozen version from our model zoo:
    ```
    wget https://nnfusion.blob.core.windows.net/models/tensorflow/frozen_lstm_l8s8h256_bs1.pb
    ```
4. First, you can try to compile this model WITHOUT kernel tuning:
    ```
    nnfusion frozen_lstm_l8s8h256_bs1.pb
    cd nnfusion_rt/cuda_codegen && cmake . && make -j
    ./main_test
    ```
    
5. By default, NNFusion select some pre-defined kernels to generate the model. Now, if you would like to further improve  your model execution performance by tuning each kernel on this device, you can add the following args:

    ```
    nnfusion frozen_lstm_l8s8h256_bs1.pb -fkernel_tuning_steps=100 -fantares_mode=true
    ```
    The `fkernel_tuning_steps=N` option allows you to tune each operator in your model for `N` steps with our Antares tuning service. By default, this tuning process is running in a blocking mode, which will usually takes for hours to finish a full model tuning. The progress is showed as below:

    ```
    Tuning [0/17 ops]: op=Concat, name=BasicLSTMCellZeroState/c..:
    [##################################################] 100%
    Tuning [1/17 ops]: op=Slice, name=Slice_195:
    [##################################################] 100%
    Tuning [2/17 ops]: op=Reshape, name=strided_slice:
    [##################################################] 100%
    Tuning [3/17 ops]: op=Slice, name=Slice_197:
    [##################################################] 100%
    Tuning [4/17 ops]: op=Slice, name=Slice_199:
    [##################################################] 100%
    Tuning [5/17 ops]: op=Slice, name=Slice_201:
    [########################                          ] 48%

    ```
    NNFusion also support tunning kernels in a non-blocking model, by specifing an Antares server with `-fantares_codegen_server` option, as below:

    ```
    nnfusion frozen_lstm_l8s8h256_bs1.pb -fkernel_tuning_steps=5 -fantares_mode=true -fantares_codegen_server=127.0.0.1:8880
    ```
    Then the compilation process is not a blocking execution, and NNFusion is only submitting each tuning task into the Antares service and reports the status. It looks like,

    ```
    [INFO] 2020-12-15T05:48:27z src/nnfusion/engine/pass/graph/kernel_tuning.cpp 92  Kernel Tuning Status:
    NOTE: the tuning progress (N/M) means that the current best kernel is searched at the N-th step of the total M steps.
    |                   OP |                       NAME |     STATUS |   PROGRESS |     PERFORMANCE |
    | --------------------------------------------------------------------------------------------- |
    |                  Dot |                 MatMul_126 |  completed |      5/5  |    0.0160695 ms |
    |                  Dot |                 MatMul_177 |  completed |      5/5  |    0.0160695 ms |
    |                  Dot |                 MatMul_179 |  completed |      5/5  |    0.0160695 ms |
    |                  Dot |                 MatMul_181 |  completed |      5/5  |    0.0160695 ms |
    |                  Dot |                 MatMul_183 |  completed |      5/5  |    0.0160695 ms |
    |                  Add |                    add_310 |  completed |      2/5  |   0.00216627 ms |
    |                  Add |                    add_312 |  completed |      2/5  |   0.00216627 ms |
    |                  Add |                    add_314 |  completed |      2/5  |   0.00216627 ms |
    |                Slice |                  Slice_207 |  submitted |      2/5  |      0.00221 ms |
    |              Reshape |            strided_slice_6 |     tuning |      0/5  |           -1 ms |
    |                Slice |                  Slice_209 |  submitted |      2/5  |   0.00230697 ms |
    |              Reshape |            strided_slice_7 |     tuning |      0/5  |           -1 ms |
    |            Broadcast | BasicLSTMCellZeroState/z.. |  submitted |      2/5  |   0.00273326 ms |
    |            Broadcast | BasicLSTMCellZeroState/z.. |     tuning |      0/5  |           -1 ms |
    |                  Dot |                     MatMul |  submitted |      1/5  |    0.0137236 ms |
    |                  Dot |                   MatMul_2 |     tuning |      0/5  |           -1 ms |
    |                  Dot |                   MatMul_4 |     tuning |      0/5  |           -1 ms |
    |                  Dot |                   MatMul_6 |     tuning |      0/5  |           -1 ms |
    |                  Dot |                  MatMul_64 |     tuning |      0/5  |           -1 ms |
    ```
    
    You can periodically run this command to check the progress. Once all the kernel tuning status is `completed`, NNFusion will continue to generate the end-to-end source code through leveraging these new generated kernels. After that, you can just compile the generated code as usual, the only difference is the kernels used.
    ```
    cd nnfusion_rt/cuda_codegen && cmake . && make -j
    ./main_test
    ```

    Note that, in this tutorial, we just set the tunning steps as only 5 for demo purpose. In practice, you need to use a large number, i.e., 1000. Then the tuning process will take a very long time to finish. 




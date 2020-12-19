## Distributed MNIST Training Example
### Install Dependencies
```sh
# install superscaler 
$ git clone msrasrg@vs-ssh.visualstudio.com:v3/msrasrg/SuperScaler/SuperScaler && cd SuperScaler  && pip install . && cd - && rm -fr SuperScaler
# install other dependencies
$ pip install torch torchvision mpi4py
```

### Prepare data
Prepare your own trainable frozen model or you can acquire from [mnist_mlp.onnx](https://to-be-replaced-here) and GPU cluster's specification file which describes the underlying topology of your model training environment.   Since the GPU cluster's specification is used by [SuperScaler](https://github.com/microsoft/SuperScaler.git), you can learn all the details from there. Or an example  [resource_pool.yaml](https://github.com/microsoft/SuperScaler#appendix-a-sample-resource_poolyaml) is provided.


### Compile
```sh
# this will compile the frozen model for 2 GPU workers doing data parallel training on the same host
$ ../../../superscaler/nnfusion_dp_single_host.sh  mnist_mlp.onnx "-f onnx -p \"batch:3\" -fautodiff -ftraining_mode -fextern_result_memory=True" localhost:2  resource_pool.yaml
```

### Build
```sh
$ cd build && cmake . > /dev/null && make -j > /dev/null || make
```

### Train
```sh
$ ./train.sh
```



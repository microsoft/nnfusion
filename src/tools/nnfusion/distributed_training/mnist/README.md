## Distributed MNIST Training Example
### Install Dependencies
```sh
# install superscaler 
$ python3 -m pip install tensorflow==1.15
$ git clone https://github.com/microsoft/SuperScaler.git && cd SuperScaler && python3 -m pip install . && cd - && rm -fr SuperScaler
# install other dependencies
$ python3 -m pip install torch torchvision mpi4py
```

### Prepare data
Prepare your own trainable frozen model or you can acquire from [mnist_mlp.onnx](https://nnfusion.blob.core.windows.net/models/onnx/mnist_mlp.onnx) and GPU cluster's specification file which describes the underlying topology of your model training environment.   Since the GPU cluster's specification is used by [SuperScaler](https://github.com/microsoft/SuperScaler.git), you can learn all the details from there. Or an example  [resource_pool.yaml](https://github.com/microsoft/SuperScaler#appendix-a-sample-resource_poolyaml) is provided.
```sh
$ cd ./src/tools/nnfusion/distributed_training/mnist
$ wget <URL of mnist_mlp.onnx>
$ wget <URL of resource_pool.yaml>
```

### Compile
```sh
# this will compile the frozen model for 2 GPU workers doing data parallel training on the same host
$ bash ../../../superscaler/nnfusion_dp_single_host.sh  mnist_mlp.onnx "-f onnx -p \"batch:3\" -fautodiff -ftraining_mode -fextern_result_memory=True" localhost:2  resource_pool.yaml
```

### Build
```sh
$ cd build && cmake . && make -j
```

### Train
```sh
$ bash ./train.sh
# in case you are using older version MPI:
$ mpirun -np 2 -x PATH -x LD_LIBRARY_PATH bash -c 'python3 nnf_py/train.py $OMPI_COMM_WORLD_LOCAL_RANK/plan.json'
```



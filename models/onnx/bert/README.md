## Build NNFusion with ONNX support
```bash
mkdir build && cd build
cmake .. -DONNX_FRONTEND=1
make -j
```

## Compile ONNX BERT
```bash
## This model contains dynamic axes, here we compile a batch 3, seq 512 model.
./src/tools/nnfusion/nnfusion ../models/onnx/bert/bert_layer24.onnx -f onnx -p "batch:3;sequence:512"
```

## Test compiled model
```bash
## By default it generates cuda code
cd nnfusion_rt/cuda_codegen
cmake .
make -j
./main_test
```




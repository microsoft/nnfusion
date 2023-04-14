## Requirements
TVM https://github.com/nox-410/tvm/tree/welder
NNFUSION https://github.com/nox-410/nnfusion/tree/welder
CUTLASS https://github.com/nox-410/cutlass/tree/welder

```bash
git clone https://github.com/nox-410/tvm --recursive -b welder
# Fill in USE_CUDA and USE_LLVM in tvm/cmake/config.cmake
# You need to install LLVM first if you don't have one.
mkdir -p tvm/build && cd tvm/build && cp ../cmake/config.cmake . && cmake .. && make -j && cd -
export PYTHONPATH="$PYTHONPATH:$PWD/tvm/python"

git clone https://github.com/nox-410/nnfusion -b welder
mkdir -p nnfusion/build && cd nnfusion/build && cmake .. && make -j && cd -
export PATH="$PATH:$PWD/nnfusion/build/src/tools/nnfusion"

git clone https://github.com/nox-410/cutlass -b welder
export CPLUS_INCLUDE_PATH="$CPLUS_INCLUDE_PATH:$PWD/cutlass/include"

pip install onnx==1.12 onnxruntime-gpu==1.12 onnxconverter_common==1.12
pip install pytorch==1.12 torchvision==0.13 timm==0.5.4
pip install attrs cloudpickle decorator psutil synr tornado xgboost
```


Finally, add ./python to PYTHONPATH.

## Usage
### Prepare onnx model

Supporting opset11, use ./testing/torch2onnx.py to get some supported models.

### Run the compiler

```bash
nnfusion model.onnx -f onnx -ftune_output_file=model.json &&
python3 -m run_compiler model.json tuned.json --device 0 --topk 20 &&
nnfusion model.onnx -f onnx -ftune_output_file=/dev/null -ftune_input_file=tuned.json &&
rm -rf nnfusion_rt/cuda_codegen/build/ && cmake -S nnfusion_rt/cuda_codegen/ -B nnfusion_rt/cuda_codegen/build/ &&
make -C nnfusion_rt/cuda_codegen/build/
```
This will extract the IR compute graph first (first line).

Then run the compiler (second line).

Compose final code and compile (third & fourth line).

### run test

```bash
cd nnfusion_rt/cuda_codegen && ./build/main_test
```

## Extra

compare end to end model correctness : ./testing/test_acc.py prefix

single operator tuning : ./testing/test_policy.py

## Requirements
TVM https://github.com/nox-410/tvm/tree/memfusion
NNFUSION https://github.com/nox-410/nnfusion/tree/smem_fuse

python==3.7

pip:
- pytorch==1.12.0
- torchvision==0.13.0
- onnx==1.12
- onnxruntime==1.12
- timm==0.5.4

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

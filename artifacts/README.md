# Installation

1. install antares
```
python3 -m pip install --upgrade antares
```
2. integrate roller into antares
```
cp default.py [path-to-antares]/antares_core/backends/c-cuda/schedule/standard/
```
3. install roller
```
python setup.py install
```
4. test roller tuning
```
BACKEND=c-cuda STEP=10 COMPUTE_V1='- S = 4096; einstein_v2(input_dict={"input0": {"dtype": "float32", "shape": [S, S]}, "input1": {"dtype": "float32", "shape": [S, S]}}, exprss="output0[N, M] +=! input0[N, K] * input1[K, M]")' antares
```
- OSDI'20 Artifact Evaluation of paper #292, titled "[Rammer: Enabling Holistic Deep Learning Compiler Optimizations with rTasks](https://www.usenix.org/conference/osdi20/presentation/ma)". 
Please refer to the [osdi20_artifact branch](https://github.com/microsoft/nnfusion/tree/osdi20_artifact/artifacts)**


- OSDI'22 Artifact Evaluation of paper #158, titled "[Roller: Fast and Efficient Tensor Compilation for Deep Learning](https://www.usenix.org/conference/osdi22/presentation/zhu)". 
Please refer to the [osdi22_artifact branch](https://github.com/microsoft/nnfusion/tree/osdi22_artifact/artifacts)**

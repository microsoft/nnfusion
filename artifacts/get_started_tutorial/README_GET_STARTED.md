# Get Started Tutorial: Generate a Matmul Kernel

We assume you already build and install Roller compiler folloing the *Environment Preparation* section in [README.md](../README.md).

The goal of this tutorial is to demonstrate how to leverage Roller to quickly generate a DNN operator kernel, e.g., Matmul.

## Input Tensor Experssion
Currently, Roller compiler takes inputs as a tensor expression implemented in TVM IR with specifying input and output tensor shapes. 
For example, below is a MatMul tensor expression example for Roller,
```
def matmul_expr(shape, dataType="float32", pad={}):
    M, N, K = shape
    A = te.placeholder((M, K), dtype=dataType, name="A")
    B = te.placeholder((K, N), dtype=dataType, name="B")
    k = te.reduce_axis((0, K), name="k")
    C = te.compute((M, N), lambda y, x: te.sum((A[y, k]) * B[k, x], axis=k), name='compute')
    return [A, B], [C]
```

For convenience, we have pre-implemented several common tensor expressions under `roller/test_config/` folder.

## Specify Architecture
Roller can generate high-performance kernel for differnet hardware architecture. User can configure and modify architecture based on their needs. Under `roller/arch/` folder we provide some common hardware configurations. 


## Compile
Under `roller/` path, we prepare a helper script named `test_op_mp.py` to quickly generate kernels given an expression. For example, the below command will generate a `[4096, 4096] x [4096, 4096]` MatMul kernel for CUDA V100 GPUs.

```
python test_op_mp.py --code_dir tmp_dir/ --smem_tiling --reg_tiling --op matmul_expr --shape 4096 4096 4096
```

By default, Roller construct top-10 kernels and profile to choose the best one. The exeuction log looks like,

```
Namespace(arch='V100', backend='tvm', code_dir='tmp_dir/', data_type='float32', eval_bar=[1, 5, 10, 20, 50], fuse=False, keep_tiny=False, num_threads=4, op='matmul_expr', padding_threshold_cap=1.0, reg_tiling=True, rtile0_shape=[64, 128, 32], rtile1_shape=[8, 8, 1], rtile2_shape=[1, 1, 1], schedule_fuse=False, shape=[4096, 4096, 4096], smem_tiling=True, st_align=False, topk=10, use_artificial_rtile=False, use_tc=False)

...

top1 time: 13.167 ms
top10 time: 10.99 ms
best idx: 5
best config: [level 0: [tile: [128, 64]; step: [32]]][level 1: [tile: [8, 4]; step: [1]]][level 2: [tile: [1, 1]; step: [1]]]
top1 compile time: 0.003381490707397461 s
top10 compile time: 13.96477484703064 s
```
It shows that the first constructed kernel's performance (e.g., kernel time) is 13.167ms and the best one among the top 10 kernels is 10.99 ms. The top-1 compilation time is 0.003s and top-10 compilation time is 13.9s (mainly spends on compiling and measuring each constructed kernels).
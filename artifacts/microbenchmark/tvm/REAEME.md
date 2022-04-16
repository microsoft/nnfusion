# 1. microbenchmark
## 1.1 ansor
tune and profile
```
bash script/op/bench0_conv2d.sh ansor
```


profile only
```
bash script/op/bench0_conv2d.sh ansor path_to_log_directory
```
## 1.2 autotvm
# 2. e2e
## 2.1 kernel generation
## 2.2 kernel injection
## 2.3 run model
# 3. scale
# 4.checklist
## 4.1 microbenchmark
ansor
[x] conv (missing last 29, tuned)
TiledCompiler/tiled-compiler/microbenchmark/tvm/ansor/conv
[x] depthwise
TiledCompiler/tiled-compiler/microbenchmark/tvm/ansor/depthwise
[x] matmul
TiledCompiler/tiled-compiler/microbenchmark/tvm/ansor/matmul
[x] elementwise
TiledCompiler/tiled-compiler/microbenchmark/tvm/ansor/elementwise
[x] pooling
TiledCompiler/tiled-compiler/microbenchmark/tvm/ansor/pooling
[x] reduction
TiledCompiler/tiled-compiler/microbenchmark/tvm/ansor/reduction

autotvm
[x] conv
TiledCompiler/tiled-compiler/microbenchmark/tvm/autotvm/conv
[x] depthwise
TiledCompiler/tiled-compiler/microbenchmark/tvm/autotvm/depthwise
[x] matmul
TiledCompiler/tiled-compiler/microbenchmark/tvm/autotvm/matmul_nt
TiledCompiler/tiled-compiler/microbenchmark/tvm/autotvm/matmul_nn
[x] elementwise (no log, no tuneable)
[x] pooling (no log, no tuneable)
[x] reduction (no log, no tuneable)

## 4.2 e2e
ansor
[x] .cc
TiledCompiler/tiled-compiler/microbenchmark/tvm/e2e/ansor
[x] .json
TiledCompiler/tiled-compiler/microbenchmark/tvm/e2e/ansor

autotvm
[x] .log
TiledCompiler/tiled-compiler/microbenchmark/tvm/e2e/autotvm_log
[x] .cc
TiledCompiler/tiled-compiler/microbenchmark/tvm/e2e/autotvm
[x] .json
TiledCompiler/tiled-compiler/microbenchmark/tvm/e2e/autotvm

## 4.3 scale out
[x] conv (C=1024, H=14, F=2048, K=1, S=2)
TiledCompiler/tiled-compiler/microbenchmark/tvm/scale_out/conv_scale
[x] matmul (K=1024, N=4096)
TiledCompiler/tiled-compiler/microbenchmark/tvm/scale_out/matmul_bert_scale
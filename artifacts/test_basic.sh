# ElementWise
# 1. transpose
# 2. add
# 3. reshape
# 4. condition relu
# 5. condition relu for dynamic data type
# 6. depthtospace
# 7. pad
# 8. tile
# 9. sigmoid
# 10. divnonan
# 11. logical bool operation
# 12. cast + range
# 13. cast + tanh + range
# 14. scalar compute

CHECK=1 BACKEND=c-cuda COMPUTE_V1='- einstein_v2("output0[N, C, H, W] = input0[N, H, W, C]", input_dict={"input0": {"dtype": "float32", "shape": [32, 229, 229, 3]}})' antares

CHECK=1 BACKEND=c-cuda COMPUTE_V1='- einstein_v2("output0[N] = input0[N] + input1[N]", input_dict={"input0": {"dtype": "float32", "shape": [1024 * 512]}, "input1": {"dtype": "float32", "shape": [1024 * 512]}})' antares

CHECK=1 BACKEND=c-cuda COMPUTE_V1='- einstein_v2("output0[A, B, C] = input0[A, B, C / 64, C % 64] where C in 128", input_dict={"input0": {"dtype": "float32", "shape": [3, 3, 2, 64]}})' antares

CHECK=1 BACKEND=c-cuda COMPUTE_V1='- einstein_v2("output0[N, C] = input0[N, C].when([input0[N, C] > 0.0], 0.0)", input_dict={"input0": {"dtype": "float32", "shape": [1024, 512]}})' antares

CHECK=1 BACKEND=c-cuda COMPUTE_V1='- einstein_v2("output0[N, C] = input0[N, C].when([input0[N, C] > const(0.0).cast(input0[N, C].dtype())], const(0.0).cast(input0[N, C].dtype()))", input_dict={"input0": {"dtype": "float32", "shape": [1024, 512]}})' antares

CHECK=1 BACKEND=c-cuda COMPUTE_V1='- einstein_v2("output0[N, H, C0, W, C1, C2] = input0[N, H, W, C0, C1, C2]", input_dict={"input0": {"dtype": "float32", "shape": [1, 256, 256, 2, 2, 4]}})' antares

CHECK=1 BACKEND=c-cuda COMPUTE_V1='- einstein_v2("output0[N, C, HO, WO] = input0[N, C, -1 + HO, -1 + WO].when([-1 + HO >= 0, -1 + HO < 32, -1 + WO >= 0, -1 + WO < 32], 0.0) where HO in 34, WO in 34", input_dict={"input0": {"dtype": "float32", "shape": [32, 3, 32, 32]}})' antares

CHECK=1 BACKEND=c-cuda COMPUTE_V1='- einstein_v2("output0[ON, OC] = input0[ON % 2, OC % 16] where ON in 1024, OC in 4096", input_dict={"input0": {"dtype": "float32", "shape": [2, 16]}})' antares

CHECK=1 BACKEND=c-cuda COMPUTE_V1='- einstein_v2("output0[N, M] = 1.0 / (1.0 + (-input0[N, M]).call(`exp`))", { "input0": {"dtype": "float32", "shape": [1024, 512]} })' antares

CHECK=1 BACKEND=c-cuda COMPUTE_V1='- einstein_v2("output0[N] = (input0[N] / input1[N]).when([input1[N] != 0], 0.0)", input_dict={"input0": {"dtype": "float32", "shape": [32 * 1024]}, "input1": {"dtype": "float32", "shape": [32 * 1024]}})' antares

BACKEND=c-cuda COMPUTE_V1='- einstein_v2("output0[N, M] = input0[N, M] & ~input1[N, M]", { "input0": {"dtype": "int8", "shape": [1024, 512]}, "input1": {"dtype": "int8", "shape": [1024, 512]} })' antares

CHECK=1 BACKEND=c-cuda COMPUTE_V1='- einstein_v2("output0[N] = N.cast(`float32`) where N in 1024", {})' antares

CHECK=1 BACKEND=c-cuda COMPUTE_V1='- einstein_v2("output0[N] = N.cast(`float32`).call(`tanh`) where N in 1024", {})' antares

CHECK=1 BACKEND=c-cuda COMPUTE_V1='- einstein_v2("output0[] = input0[] + input1[]", input_dict={"input0": {"dtype": "float32", "shape": []}, "input1": {"dtype": "float32", "shape": []}})' antares

echo "Finish Elementwise\n"

# Data Movement
# 1. slice
# 2. take
# 3. gather

CHECK=1 BACKEND=c-cuda COMPUTE_V1='- einstein_v2("output0[N, F] = input0[N, F, 2]", input_dict={"input0": {"dtype": "float32", "shape": [1, 16, 32]}})' antares

CHECK=1 BACKEND=c-cuda COMPUTE_V1='- einstein_v2("output0[F, C] = input0[input1[F], C]", input_dict={"input0": {"dtype": "float32", "shape": [30528, 1024]}, "input1": {"dtype": "int32", "shape": [3072]}})' antares

CHECK=1 BACKEND=c-cuda COMPUTE_V1='- einstein_v2("output0[N, F] = input0[input1[N, F]]", input_dict={"input0": {"dtype": "float32", "shape": [65536]}, "input1": {"dtype": "int32", "shape": [4, 64]}})' antares

echo "Finish Data Movement\n"

# MatMul
# 1. matmul 4096 x 4096 x 4096
# 2. matmul 65536 x 30522 x 1024
# 3. matmul bias add
# 4. batched matmul

CHECK=1 BACKEND=c-cuda COMPUTE_V1='- S = 4096; einstein_v2(input_dict={"input0": {"dtype": "float32", "shape": [S, S]}, "input1": {"dtype": "float32", "shape": [S, S]}}, exprss="output0[N, M] +=! input0[N, K] * input1[K, M]")' antares

CHECK=1 BACKEND=c-cuda COMPUTE_V1='- einstein_v2(input_dict={"input0": {"dtype": "float32", "shape": [65536, 30522]}, "input1": {"dtype": "float32", "shape": [30522, 1024]}}, exprss="output0[N, M] +=! input0[N, K] * input1[K, M]")' antares

CHECK=1 BACKEND=c-cuda COMPUTE_V1='- einstein_v2("output0[N, M] +=! input0[N, K] * input1[K, M] + input2[M] / K.val()", { "input0": {"dtype": "float32", "shape": [1024, 512]}, "input1": {"dtype": "float32", "shape": [512, 512]}, "input2": {"dtype": "float32", "shape": [512]} })' antares

CHECK=1 BACKEND=c-cuda COMPUTE_V1='- einstein_v2("output0[B, N, M] +=! input0[B, N, K] * input1[B, K, M]", input_dict={"input0": {"dtype": "float32", "shape": [3, 1024, 512]}, "input1": {"dtype": "float32", "shape": [3, 512, 512]}})' antares

echo "Finish MatMul\n"

# Pool
# 1. maxpool
# 2. avgpool

CHECK=1 BACKEND=c-cuda COMPUTE_V1='- einstein_v2("output0[N, C, HO, WO] >=! input0[N, C, HO * 2 + KH, WO * 2 + KW] where HO in 6, WO in 6, KW in 2, KH in 2", input_dict={"input0": {"dtype": "float32", "shape": [32, 3, 12, 12]}})' antares

CHECK=1 BACKEND=c-cuda COMPUTE_V1='- einstein_v2("output0[NC, HO, WO] +=! input0[NC, HO * 3 + KH, WO * 3 + KW] / 9.0 where HO in 85, WO in 85, KW in 3, KH in 3", input_dict={"input0": {"dtype": "float32", "shape": [1024, 255, 255]}})' antares

echo "Finish Pool\n"

# Reduce
# 1. ReduceSum
# 2. ReduceMin
# 3. single reduction

CHECK=1 BACKEND=c-cuda COMPUTE_V1='- einstein_v2("output0[N] +=! input0[N, C]", input_dict={"input0": {"dtype": "float32", "shape": [32, 1024]}})' antares

CHECK=1 BACKEND=c-cuda COMPUTE_V1='- einstein_v2("output0[N] +=! input0[N, C]", input_dict={"input0": {"dtype": "float32", "shape": [32, 1024]}})' antares

CHECK=1 BACKEND=c-cuda COMPUTE_V1='- einstein_v2("output0[] +=! input0[N] * input1[N]", input_dict={"input0": {"dtype": "float32", "shape": [1024]}, "input1": {"dtype": "float32", "shape": [1024]}})' antares

echo "Finish Reduce\n"

# Direct Conv
# 1. S1D1P0
# 2. S2D1P0
# 3. S1D1P0
# 4. DepthwiseConv

CHECK=1 BACKEND=c-cuda COMPUTE_V1='- einstein_v2("output0[N, F, HO, WO] +=! input0[N, C, HO + KH, WO + KW] * input1[F, C, KH, KW] where HO in 30, WO in 30", { "input0": {"dtype": "float32", "shape": [16, 64, 32, 32]}, "input1": {"dtype": "float32", "shape": [256, 64, 3, 3]}})' antares

CHECK=1 BACKEND=c-cuda COMPUTE_V1='- einstein_v2("output0[N, F, HO, WO] +=! input0[N, C, HO * 2 + KH, WO * 2 + KW] * input1[F, C, KH, KW] where HO in 165, WO in 165", { "input0": {"dtype": "float32", "shape": [128, 3, 332, 332]}, "input1": {"dtype": "float32", "shape": [96, 3, 3, 3]}})' antares

CHECK=1 BACKEND=c-cuda COMPUTE_V1='- _N, _CI, _H, _W, _CO, _KH, _KW, _SH, _SW, _PH, _PW = 16, 64, 32, 32, 256, 3, 3, 1, 1, 0, 0; _HO, _WO = (_H - _KH + _PH * 2) // _SH + 1, (_W - _KW + _PW * 2) // _SW + 1; einstein_v2(f"output0[N, F, HO, WO] +=! input0[N, C, HO * {_SH} + KH - {_PH}, WO * {_SW} + KW - {_PW}].when([HO * {_SH} + KH - {_PH} >= 0, HO * {_SH} + KH - {_PH} < {_H}, WO * {_SW} + KW - {_PW} >= 0, WO * {_SW} + KW - {_PW} < {_W}], 0.0) * input1[F, C, KH, KW] where HO in {_HO}, WO in {_WO}", { "input0": {"dtype": "float32", "shape": [_N, _CI, _H, _W]}, "input1": {"dtype": "float32", "shape": [_CO, _CI, _KH, _KW]}})' antares

CHECK=1 BACKEND=c-cuda COMPUTE_V1='- einstein_v2("output0[N, C, HO, WO] +=! input0[N, C, HO + KH, WO + KW] * input1[KH, KW, C, 0] where HO in 30, WO in 30", input_dict={"input0": {"dtype": "float32", "shape": [32, 16, 32, 32]}, "input1": {"dtype": "float32", "shape": [3, 3, 16, 1]}})' antares

echo "Finish Direct Conv\n"

# Fusion
# 1. addmatmul head fusion

CHECK=1 BACKEND=c-cuda COMPUTE_V1='- einstein_v2("temp0[K, N] = input0[N, K] + 100; output0[N, M] +=! temp0[K, N] * input1[K, M] where K in 10", { "input0": {"dtype": "float32", "shape": [1024, 512]}, "input1": {"dtype": "float32", "shape": [512, 512]}})' antares

echo "Finish Fusion\n"

# Other Utilities
# 1. small sized elementwise
# 2. onehot

CHECK=1 BACKEND=c-cuda COMPUTE_V1='- einstein_v2("output0[N, F] = input0[N, F]", input_dict={"input0": {"dtype": "float32", "shape": [1, 16]}})' antares

CHECK=1 BACKEND=c-cuda COMPUTE_V1='- einstein_v2("output0[N, F] = const(1.0).when([input0[N] == F], const(0.0)) where F in 128", input_dict={"input0": {"dtype": "int32", "shape": [4]}})' antares

echo "Finish Other Utilities\n"

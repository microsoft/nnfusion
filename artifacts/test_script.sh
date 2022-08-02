# MATMUL

BACKEND=c-cuda COMPUTE_V1='- S = 4096; einstein_v2(input_dict={"input0": {"dtype": "float32", "shape": [S, S]}, "input1": {"dtype": "float32", "shape": [S, S]}}, exprss="output0[N, M] +=! input0[N, K] * input1[K, M]")' antares

BACKEND=c-cuda COMPUTE_V1='- einstein_v2(input_dict={"input0": {"dtype": "float32", "shape": [65536, 30522]}, "input1": {"dtype": "float32", "shape": [30522, 1024]}}, exprss="output0[N, M] +=! input0[N, K] * input1[K, M]")' antares

# MatMulBiasAdd

BACKEND=c-cuda COMPUTE_V1='- einstein_v2("output0[N, M] +=! input0[N, K] * input1[K, M] + input2[M] / K.val()", { "input0": {"dtype": "float32", "shape": [1024, 512]}, "input1": {"dtype": "float32", "shape": [512, 512]}, "input2": {"dtype": "float32", "shape": [512]} })' antares

# BATCHED MATMUL

BACKEND=c-cuda COMPUTE_V1='- einstein_v2("output0[B, N, M] +=! input0[B, N, K] * input1[B, K, M]", input_dict={"input0": {"dtype": "float32", "shape": [3, 1024, 512]}, "input1": {"dtype": "float32", "shape": [3, 512, 512]}})' antares

# TODO: Multiple Outpus

# BACKEND=c-cuda COMPUTE_V1='- einstein_v2("output0[N] = input0[N] + input1[N]; output1[N] = input0[N] * 2; output2[N] = input1[N] + output1[N];", input_dict={"input0": {"dtype": "float32", "shape": [1024 * 512]}, "input1": {"dtype": "float32", "shape": [1024 * 512]}}, extra_outputs=["output0", "output1", "output2"])' antares

# Transpose need change variable val

# BACKEND=c-cuda COMPUTE_V1='- einstein_v2("output0[N, C, H, W] = input0[N, H, W, C]", input_dict={"input0": {"dtype": "float32", "shape": [32, 229, 229, 3]}})' antares

# POOL

# maxpool
BACKEND=c-cuda COMPUTE_V1='- einstein_v2("output0[N, C, HO, WO] >=! input0[N, C, HO * 2 + KH, WO * 2 + KW] where HO in 6, WO in 6, KW in 2, KH in 2", input_dict={"input0": {"dtype": "float32", "shape": [32, 3, 12, 12]}})' antares

# avgpool
BACKEND=c-cuda COMPUTE_V1='- einstein_v2("output0[NC, HO, WO] +=! input0[NC, HO * 3 + KH, WO * 3 + KW] / 9.0 where HO in 85, WO in 85, KW in 3, KH in 3", input_dict={"input0": {"dtype": "float32", "shape": [1024, 255, 255]}})' antares

# REDUCE

# reduce sum
BACKEND=c-cuda COMPUTE_V1='- einstein_v2("output0[N] +=! input0[N, C]", input_dict={"input0": {"dtype": "float32", "shape": [32, 1024]}})' antares

# reduce min
BACKEND=c-cuda COMPUTE_V1='- einstein_v2("output0[N] <=! input0[N, C]", input_dict={"input0": {"dtype": "float32", "shape": [32, 1024]}})' antares

# CONV

# S1D1P0
BACKEND=c-cuda COMPUTE_V1='- einstein_v2("output0[N, F, HO, WO] +=! input0[N, C, HO + KH, WO + KW] * input1[F, C, KH, KW] where HO in 30, WO in 30", { "input0": {"dtype": "float32", "shape": [16, 64, 32, 32]}, "input1": {"dtype": "float32", "shape": [256, 64, 3, 3]}})' antares

# S2D1P0
BACKEND=c-cuda COMPUTE_V1='- einstein_v2("output0[N, F, HO, WO] +=! input0[N, C, HO * 2 + KH, WO * 2 + KW] * input1[F, C, KH, KW] where HO in 165, WO in 165", { "input0": {"dtype": "float32", "shape": [128, 3, 332, 332]}, "input1": {"dtype": "float32", "shape": [96, 3, 3, 3]}})' antares

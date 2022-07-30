# matmul
BACKEND=c-cuda COMPUTE_V1='- S = 4096; einstein_v2(input_dict={"input0": {"dtype": "float32", "shape": [S, S]}, "input1": {"dtype": "float32", "shape": [S, S]}}, exprss="output0[N, M] +=! input0[N, K] * input1[K, M]")' antares

BACKEND=c-cuda COMPUTE_V1='- einstein_v2(input_dict={"input0": {"dtype": "float32", "shape": [65536, 30522]}, "input1": {"dtype": "float32", "shape": [30522, 1024]}}, exprss="output0[N, M] +=! input0[N, K] * input1[K, M]")' antares

# batched matmul

COMPUTE_V1='- einstein_v2("output0[B, N, M] +=! input0[B, N, K] * input1[B, K, M]", input_dict={"input0": {"dtype": "float32", "shape": [3, 1024, 512]}, "input1": {"dtype": "float32", "shape": [3, 512, 512]}})' antares

# pool

# maxpool

COMPUTE_V1='- einstein_v2("output0[N, C, HO, WO] >=! input0[N, C, HO * 2 + KH, WO * 2 + KW] where HO in 6, WO in 6, KW in 2, KH in 2", input_dict={"input0": {"dtype": "float32", "shape": [32, 3, 12, 12]}})' antares

# avgpool
COMPUTE_V1='- einstein_v2("output0[NC, HO, WO] +=! input0[NC, HO * 3 + KH, WO * 3 + KW] / 9.0 where HO in 85, WO in 85, KW in 3, KH in 3", input_dict={"input0": {"dtype": "float32", "shape": [1024, 255, 255]}})' antares

# conv

COMPUTE_V1='- einstein_v2("output0[N, F, HO, WO] +=! input0[N, C, HO + KH, WO + KW] * input1[F, C, KH, KW] where HO in 30, WO in 30", { "input0": {"dtype": "float32", "shape": [16, 64, 32, 32]}, "input1": {"dtype": "float32", "shape": [256, 64, 3, 3]}})' antares

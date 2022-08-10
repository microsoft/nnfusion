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

CHECK=1 BACKEND=c-cuda COMPUTE_V1='- einstein_v2("output0[N, C, H, W] = input0[N, H, W, C]", input_dict={"input0": {"dtype": "float32", "shape": [32, 229, 229, 3]}})' antares

CHECK=1 BACKEND=c-cuda COMPUTE_V1='- einstein_v2("output0[N] = input0[N] + input1[N]", input_dict={"input0": {"dtype": "float32", "shape": [1024 * 512]}, "input1": {"dtype": "float32", "shape": [1024 * 512]}})' antares

CHECK=1 BACKEND=c-cuda COMPUTE_V1='- einstein_v2("output0[A, B, C] = input0[A, B, C / 64, C % 64] where C in 128", input_dict={"input0": {"dtype": "float32", "shape": [3, 3, 2, 64]}})' antares

CHECK=1 BACKEND=c-cuda COMPUTE_V1='- einstein_v2("output0[N, C] = input0[N, C].when([input0[N, C] > 0.0], 0.0)", input_dict={"input0": {"dtype": "float32", "shape": [1024, 512]}})' antares

CHECK=1 BACKEND=c-cuda COMPUTE_V1='- einstein_v2("output0[N, C] = input0[N, C].when([input0[N, C] > const(0.0).cast(input0[N, C].dtype())], const(0.0).cast(input0[N, C].dtype()))", input_dict={"input0": {"dtype": "float32", "shape": [1024, 512]}})' antares

CHECK=1 BACKEND=c-cuda COMPUTE_V1='- einstein_v2("output0[N, H, C0, W, C1, C2] = input0[N, H, W, C0, C1, C2]", input_dict={"input0": {"dtype": "float32", "shape": [1, 256, 256, 2, 2, 4]}})' antares

CHECK=1 BACKEND=c-cuda COMPUTE_V1='- einstein_v2("output0[N, C, HO, WO] = input0[N, C, -1 + HO, -1 + WO].when([-1 + HO >= 0, -1 + HO < 32, -1 + WO >= 0, -1 + WO < 32], 0.0) where HO in 34, WO in 34", input_dict={"input0": {"dtype": "float32", "shape": [32, 3, 32, 32]}})' antares

CHECK=1 BACKEND=c-cuda COMPUTE_V1='- einstein_v2("output0[ON, OC] = input0[ON % 2, OC % 16] where ON in 1024, OC in 4096", input_dict={"input0": {"dtype": "float32", "shape": [2, 16]}})' antares

CHECK=1 BACKEND=c-cuda COMPUTE_V1='- einstein_v2("output0[N, M] = 1.0 / (1.0 + (-input0[N, M]).call(`exp`))", { "input0": {"dtype": "float32", "shape": [1024, 512]} })' antares

echo "Finish Elementwise\n"

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

CHECK=1 BACKEND=c-cuda COMPUTE_V1='- einstein_v2("output0[N] +=! input0[N, C]", input_dict={"input0": {"dtype": "float32", "shape": [32, 1024]}})' antares

CHECK=1 BACKEND=c-cuda COMPUTE_V1='- einstein_v2("output0[N] +=! input0[N, C]", input_dict={"input0": {"dtype": "float32", "shape": [32, 1024]}})' antares

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

# Implicit GEMM
# 1. Depthwise Conv S2D1P2 128 84 42 42 5 5
# 2. Depthwise Conv S1D1P2 128 42 83 83 5 5
# 3. Depthwise Conv S1D1P2 128 336 21 21 5 5
# 4. Depthwise Conv S2D1P2 128 42 83 83 5 5
# 5. Depthwise Conv S2D1P3 128 84 42 42 7 7
# 6. Depthwise Conv S1D1P1 128 672 11 11 3 3
# 7. Depthwise Conv S1D1P3 128 42 83 83 7 7

CHECK=1 BACKEND=c-cuda COMPUTE_V1='- _N, _C, _H, _W, _KH, _KW, _SH, _SW, _PH, _PW = 128, 84, 83, 83, 5, 5, 2, 2, 2, 2; \
              _HO, _WO = (_H - _KH + _PH * 2) // _SH + 1, (_W - _KW + _PW * 2) // _SW + 1; \
              einstein_v2(f" \
                data_pad[N, C, H, W] = data[N, C, H-{_PH}, W-{_PW}].when([{_PH} <= H, H < {_H+_PH}, {_PW} <= W, W < {_W+_PW}], 0.0) where N in {_N}, C in {_C}, H in {_H + 2 * _PH}, W in {_W + 2 * _PW}; \
                kernel_pad[C, KH, KW] = kernel[C, KH, KW] where C in {_C}, KH in {_KH}, KW in {_KW}; \
                depthwiseconv2d[N, C, H, W] +=! data_pad[N, C, H * {_SH} + KH, W * {_SW} + KW] * kernel_pad[C, KH, KW] where N in {_N}, C in {_C}, H in {_HO}, W in {_WO}, KH in {_KH}, KW in {_KW}; \
                depthwiseconv2d_unpad[N, C, H, W] = depthwiseconv2d[N, C, H, W] where N in {_N}, C in {_C}, H in {_HO}, W in {_WO} \
              ", { "data": {"dtype": "float32", "shape": [_N, _C, _H, _W]}, "kernel": {"dtype": "float32", "shape": [_C, _KH, _KW]}})' antares

CHECK=1 BACKEND=c-cuda COMPUTE_V1='- _N, _C, _H, _W, _KH, _KW, _SH, _SW, _PH, _PW = 128, 42, 83, 83, 5, 5, 1, 1, 2, 2; \
              _HO, _WO = (_H - _KH + _PH * 2) // _SH + 1, (_W - _KW + _PW * 2) // _SW + 1; \
              einstein_v2(f" \
                data_pad[N, C, H, W] = data[N, C, H-{_PH}, W-{_PW}].when([{_PH} <= H, H < {_H+_PH}, {_PW} <= W, W < {_W+_PW}], 0.0) where N in {_N}, C in {_C}, H in {_H + 2 * _PH}, W in {_W + 2 * _PW}; \
                kernel_pad[C, KH, KW] = kernel[C, KH, KW] where C in {_C}, KH in {_KH}, KW in {_KW}; \
                depthwiseconv2d[N, C, H, W] +=! data_pad[N, C, H * {_SH} + KH, W * {_SW} + KW] * kernel_pad[C, KH, KW] where N in {_N}, C in {_C}, H in {_HO}, W in {_WO}, KH in {_KH}, KW in {_KW}; \
                depthwiseconv2d_unpad[N, C, H, W] = depthwiseconv2d[N, C, H, W] where N in {_N}, C in {_C}, H in {_HO}, W in {_WO} \
              ", { "data": {"dtype": "float32", "shape": [_N, _C, _H, _W]}, "kernel": {"dtype": "float32", "shape": [_C, _KH, _KW]}})' antares

CHECK=1 BACKEND=c-cuda COMPUTE_V1='- _N, _C, _H, _W, _KH, _KW, _SH, _SW, _PH, _PW = 128, 336, 21, 21, 5, 5, 1, 1, 2, 2; \
              _HO, _WO = (_H - _KH + _PH * 2) // _SH + 1, (_W - _KW + _PW * 2) // _SW + 1; \
              einstein_v2(f" \
                data_pad[N, C, H, W] = data[N, C, H-{_PH}, W-{_PW}].when([{_PH} <= H, H < {_H+_PH}, {_PW} <= W, W < {_W+_PW}], 0.0) where N in {_N}, C in {_C}, H in {_H + 2 * _PH}, W in {_W + 2 * _PW}; \
                kernel_pad[C, KH, KW] = kernel[C, KH, KW] where C in {_C}, KH in {_KH}, KW in {_KW}; \
                depthwiseconv2d[N, C, H, W] +=! data_pad[N, C, H * {_SH} + KH, W * {_SW} + KW] * kernel_pad[C, KH, KW] where N in {_N}, C in {_C}, H in {_HO}, W in {_WO}, KH in {_KH}, KW in {_KW}; \
                depthwiseconv2d_unpad[N, C, H, W] = depthwiseconv2d[N, C, H, W] where N in {_N}, C in {_C}, H in {_HO}, W in {_WO} \
              ", { "data": {"dtype": "float32", "shape": [_N, _C, _H, _W]}, "kernel": {"dtype": "float32", "shape": [_C, _KH, _KW]}})' antares

CHECK=1 BACKEND=c-cuda COMPUTE_V1='- _N, _C, _H, _W, _KH, _KW, _SH, _SW, _PH, _PW = 128, 42, 165, 165, 5, 5, 2, 2, 2, 2; \
              _HO, _WO = (_H - _KH + _PH * 2) // _SH + 1, (_W - _KW + _PW * 2) // _SW + 1; \
              einstein_v2(f" \
              data_pad[N, C, H, W] = data[N, C, H-{_PH}, W-{_PW}].when([{_PH} <= H, H < {_H+_PH}, {_PW} <= W, W < {_W+_PW}], 0.0) where N in {_N}, C in {_C}, H in {_H + 2 * _PH}, W in {_W + 2 * _PW}; \
                                                              kernel_pad[C, KH, KW] = kernel[C, KH, KW] where C in {_C}, KH in {_KH}, KW in {_KW}; \
              depthwiseconv2d[N, C, H, W] +=! data_pad[N, C, H * {_SH} + KH, W * {_SW} + KW] * kernel_pad[C, KH, KW] where N in {_N}, C in {_C}, H in {_HO}, W in {_WO}, KH in {_KH}, KW in {_KW}; \
              depthwiseconv2d_unpad[N, C, H, W] = depthwiseconv2d[N, C, H, W] where N in {_N}, C in {_C}, H in {_HO}, W in {_WO} \
            ", { "data": {"dtype": "float32", "shape": [_N, _C, _H, _W]}, "kernel": {"dtype": "float32", "shape": [_C, _KH, _KW]}})' antares

CHECK=1 BACKEND=c-cuda COMPUTE_V1='- _N, _C, _H, _W, _KH, _KW, _SH, _SW, _PH, _PW = 128, 84, 83, 83, 5, 5, 2, 2, 3, 3; \
              _HO, _WO = (_H - _KH + _PH * 2) // _SH + 1, (_W - _KW + _PW * 2) // _SW + 1; \
              einstein_v2(f" \
                data_pad[N, C, H, W] = data[N, C, H-{_PH}, W-{_PW}].when([{_PH} <= H, H < {_H+_PH}, {_PW} <= W, W < {_W+_PW}], 0.0) where N in {_N}, C in {_C}, H in {_H + 2 * _PH}, W in {_W + 2 * _PW}; \
                kernel_pad[C, KH, KW] = kernel[C, KH, KW] where C in {_C}, KH in {_KH}, KW in {_KW}; \
                depthwiseconv2d[N, C, H, W] +=! data_pad[N, C, H * {_SH} + KH, W * {_SW} + KW] * kernel_pad[C, KH, KW] where N in {_N}, C in {_C}, H in {_HO}, W in {_WO}, KH in {_KH}, KW in {_KW}; \
                depthwiseconv2d_unpad[N, C, H, W] = depthwiseconv2d[N, C, H, W] where N in {_N}, C in {_C}, H in {_HO}, W in {_WO} \
              ", { "data": {"dtype": "float32", "shape": [_N, _C, _H, _W]}, "kernel": {"dtype": "float32", "shape": [_C, _KH, _KW]}})' antares

CHECK=1 BACKEND=c-cuda COMPUTE_V1='- _N, _C, _H, _W, _KH, _KW, _SH, _SW, _PH, _PW = 128, 672, 11, 11, 3, 3, 1, 1, 1, 1; \
              _HO, _WO = (_H - _KH + _PH * 2) // _SH + 1, (_W - _KW + _PW * 2) // _SW + 1; \
              einstein_v2(f" \
                data_pad[N, C, H, W] = data[N, C, H-{_PH}, W-{_PW}].when([{_PH} <= H, H < {_H+_PH}, {_PW} <= W, W < {_W+_PW}], 0.0) where N in {_N}, C in {_C}, H in {_H + 2 * _PH}, W in {_W + 2 * _PW}; \
                kernel_pad[C, KH, KW] = kernel[C, KH, KW] where C in {_C}, KH in {_KH}, KW in {_KW}; \
                depthwiseconv2d[N, C, H, W] +=! data_pad[N, C, H * {_SH} + KH, W * {_SW} + KW] * kernel_pad[C, KH, KW] where N in {_N}, C in {_C}, H in {_HO}, W in {_WO}, KH in {_KH}, KW in {_KW}; \
                depthwiseconv2d_unpad[N, C, H, W] = depthwiseconv2d[N, C, H, W] where N in {_N}, C in {_C}, H in {_HO}, W in {_WO} \
              ", { "data": {"dtype": "float32", "shape": [_N, _C, _H, _W]}, "kernel": {"dtype": "float32", "shape": [_C, _KH, _KW]}})' antares

CHECK=1 BACKEND=c-cuda COMPUTE_V1='- _N, _C, _H, _W, _KH, _KW, _SH, _SW, _PH, _PW = 128, 42, 83, 83, 7, 7, 1, 1, 3, 3; \
              _HO, _WO = (_H - _KH + _PH * 2) // _SH + 1, (_W - _KW + _PW * 2) // _SW + 1; \
              einstein_v2(f" \
                data_pad[N, C, H, W] = data[N, C, H-{_PH}, W-{_PW}].when([{_PH} <= H, H < {_H+_PH}, {_PW} <= W, W < {_W+_PW}], 0.0) where N in {_N}, C in {_C}, H in {_H + 2 * _PH}, W in {_W + 2 * _PW}; \
                kernel_pad[C, KH, KW] = kernel[C, KH, KW] where C in {_C}, KH in {_KH}, KW in {_KW}; \
                depthwiseconv2d[N, C, H, W] +=! data_pad[N, C, H * {_SH} + KH, W * {_SW} + KW] * kernel_pad[C, KH, KW] where N in {_N}, C in {_C}, H in {_HO}, W in {_WO}, KH in {_KH}, KW in {_KW}; \
                depthwiseconv2d_unpad[N, C, H, W] = depthwiseconv2d[N, C, H, W] where N in {_N}, C in {_C}, H in {_HO}, W in {_WO} \
              ", { "data": {"dtype": "float32", "shape": [_N, _C, _H, _W]}, "kernel": {"dtype": "float32", "shape": [_C, _KH, _KW]}})' antares

echo "Finish Implicit GEMM\n"

# TODO

# fused conv expr S1D1P0
# COMPUTE_V1='- _N, _CI, _H, _W, _CO, _KH, _KW, _SH, _SW, _PH, _PW = 16, 64, 32, 32, 256, 3, 3, 1, 1, 0, 0; \
#             _HO, _WO = (_H - _KH + _PH * 2) // _SH + 1, (_W - _KW + _PW * 2) // _SW + 1; \
#             _NN, _MM, _KK = _CO, _N * _HO * _WO, _CI * _KH * _KW; \
#             einstein_v2(f" \
#               data_pad[K, M] = data[M // {_KH * _KW}, K // {_KH * _KW}, M % {_HO * _WO} // {_WO} + K % {_KH * _KW} // {_KW}, M % {_HO * _WO} % {_WO} + K % {_KH * _KW} % {_KW}] where K in {_KK}, M in {_MM}; \
#               kernel_pad[N, K] = kernel[N, K // {_KH * _KW}, (K % {_KH * _KW}) // {_KW}, (K % {_KH * _KW}) % {_KW}] where N in {_NN}, K in {_KK}; \
#               output0[N, M] +=! kernel_pad[N, K] * data_pad[K, M] where N in {_NN}, M in {_MM}, K in {_KK} \
#               ", { "data": {"dtype": "float32", "shape": [_N, _CI, _H, _W]}, "kernel": {"dtype": "float32", "shape": [_CO, _CI, _KH, _KW]}})' antares

# depthwiseconv S2D1P2
# COMPUTE_V1='- _N, _C, _H, _W, _KH, _KW, _SH, _SW, _PH, _PW = 128, 84, 42, 42, 5, 5, 2, 2, 2, 2; \
#               _HO, _WO = (_H - _KH + _PH * 2) // _SH + 1, (_W - _KW + _PW * 2) // _SW + 1; \
#               einstein_v2(f" \
#                 data_pad[N, C, H, W] = data[N, C, H-{_PH}, W-{_PW}].when([{_PH} <= H, H < {_H+_PH}, {_PW} <= W, W < {_W+_PW}], 0.0) where N in {_N}, C in {_C}, H in {_H + 2 * _PH}, W in {_W + 2 * _PW}; \
#                 kernel_pad[C, KH, KW] = kernel[C, KH, KW] where C in {_C}, KH in {_KH}, KW in {_KW}; \
#                 depthwiseconv2d[N, C, H, W] +=! data_pad[N, C, H * {_SH} + KH, W * {_SW} + KW] * kernel_pad[C, KH, KW] where N in {_N}, C in {_C}, H in {_HO}, W in {_WO}, KH in {_KH}, KW in {_KW}; \
#                 depthwiseconv2d_unpad[N, C, H, W] = depthwiseconv2d[N, C, H, W] where N in {_N}, C in {_C}, H in {_HO}, W in {_WO} \
#               ", { "data": {"dtype": "float32", "shape": [_N, _C, _H, _W]}, "kernel": {"dtype": "float32", "shape": [_C, _KH, _KW]}})' antares

# Scatter4D
# COMPUTE_V1='- _B, _M = 2, 8; einstein_v2("data[indices[B, 0], indices[B, 1], indices[B, 2], indices[B, 3], M] =. updates[B, M]", input_dict={"data": {"dtype": "float32", "shape": [32, 32, 32, 32, _M]}, "indices": {"dtype": "int32", "shape": [_B, 4]}, "updates": {"dtype": "float32", "shape": [_B, _M]}})' antares

# Logical Bool Operation
# COMPUTE_V1='- einstein_v2("output0[N, M] = input0[N, M] & ~input1[N, M]", { "input0": {"dtype": "int8", "shape": [1024, 512]}, "input1": {"dtype": "int8", "shape": [1024, 512]} })' antares

# BatchnormInference
# COMPUTE_V1='- einstein_v2("output0[N, C, H, W] = bias[C] + scale[C] * (input0[N, C, H, W] - mean[C]) / (1e-5 + variance[C]).call(`sqrt`)", input_dict={"input0": {"dtype": "float32", "shape": [16, 256, 16, 16]}, "mean": {"dtype": "float32", "shape": [256]}, "variance": {"dtype": "float32", "shape": [256]}, "scale": {"dtype": "float32", "shape": [256]}, "bias": {"dtype": "float32", "shape": [256]} })' antares

# Softmax
# COMPUTE_V1='- einstein_v2("temp0[N] >=! input0[N, C]; temp1[N] +=! (input0[N, C] - temp0[N]).call(`exp`); output0[N, C] = (input0[N, C] - temp0[N]).call(`exp`) / temp1[N]", { "input0": {"dtype": "float32", "shape": [32, 1024]} })' antares

# Slice
# COMPUTE_V1='- einstein_v2("output0[N, F] = input0[N, F, 2]", input_dict={"input0": {"dtype": "float32", "shape": [1, 16, 32]}})' antares

# Concat
# COMPUTE_V1='- einstein_v2("output0[N, F] = input0[N, F].when([F < 128], input1[N, F - 128]) where F in 256", input_dict={"input0": {"dtype": "float32", "shape": [4, 128]}, "input1": {"dtype": "float32", "shape": [4, 128]}})' antares

# OneHot
# COMPUTE_V1='- einstein_v2("output0[N, F] = const(1.0).when([input0[N] == F], const(0.0)) where F in 128", input_dict={"input0": {"dtype": "int32", "shape": [4]}})' antares

# Take
# COMPUTE_V1='- einstein_v2("output0[F, C] = input0[input1[F], C]", input_dict={"input0": {"dtype": "float32", "shape": [30528, 1024]}, "input1": {"dtype": "int32", "shape": [3072]}})' antares

# Gather
# COMPUTE_V1='- einstein_v2("output0[N, F] = input0[input1[N, F]]", input_dict={"input0": {"dtype": "float32", "shape": [65536]}, "input1": {"dtype": "int32", "shape": [4, 64]}})' antares

# DivNoNan
# COMPUTE_V1='- einstein_v2("output0[N] = (input0[N] / input1[N]).when([input1[N] != 0], 0.0)", input_dict={"input0": {"dtype": "float32", "shape": [32 * 1024]}, "input1": {"dtype": "float32", "shape": [32 * 1024]}})' antares

# Broadcast
# COMPUTE_V1='- einstein_v2("output0[N, F, HO, WO] = input0[N] where F in 32, HO in 2, WO in 2", input_dict={"input0": {"dtype": "float32", "shape": [16]}})' antares

# BroadcastAll
# COMPUTE_V1='- einstein_v2("output0[N, F, HO, WO] = input0[0] where N in 8, F in 32, HO in 2, WO in 2", input_dict={"input0": {"dtype": "float32", "shape": [1]}})' antares

# Scalar Compute
# COMPUTE_V1='- einstein_v2("output0[] = input0[] + input1[]", input_dict={"input0": {"dtype": "float32", "shape": []}, "input1": {"dtype": "float32", "shape": []}})' antares

# Multiple Outpus
# BACKEND=c-cuda COMPUTE_V1='- einstein_v2("output0[N] = input0[N] + input1[N]; output1[N] = input0[N] * 2; output2[N] = input1[N] + output1[N];", input_dict={"input0": {"dtype": "float32", "shape": [1024 * 512]}, "input1": {"dtype": "float32", "shape": [1024 * 512]}}, extra_outputs=["output0", "output1", "output2"])' antares

# Cast
# COMPUTE_V1='- einstein_v2("output0[N] = N.cast(`float32`) where N in 1024", {})' antares

# Range + Tanh using External Function
# COMPUTE_V1='- einstein_v2("output0[N] = N.cast(`float32`).call(`tanh`) where N in 1024", {})' antares

# ConvBiasRelu Tail Fusion
# COMPUTE_V1='- einstein_v2("conv_out[N, F, HO, WO] +=! input0[N, C, HO + KH, WO + KW] * input1[KH, KW, C, F] where HO in 256, WO in 256; conv_bias[N, F, HO, WO] = conv_out[N, F, HO, WO] + input2[0, 0, 0, F]; output0[N, F, HO, WO] = conv_bias[N, F, HO, WO].when(conv_bias[N, F, HO, WO] > 0.0, 0.0)", input_dict={"input0": {"dtype": "float32", "shape": [1, 16, 256, 256]}, "input1": {"dtype": "float32", "shape": [1, 1, 16, 16]}, "input2": {"dtype": "float32", "shape": [1, 1, 1, 16]}})'
# antares

# ConvolutionWithPad (Fused reduce axis)

# COMPUTE_V1='- _N, _CI, _H, _W, _CO, _KH, _KW, _SH, _SW, _PH, _PW = 16, 64, 32, 32, 256, 3, 3, 1, 1, 0, 0; \
#     _HO, _WO = (_H - _KH + _PH * 2) // _SH + 1, (_W - _KW + _PW * 2) // _SW + 1; \
#       einstein_v2(f" \
#           output0[N, F, HO, WO] +=! input0[N, CKHKW // {_KW * _KH}, HO * {_SH} + ((CKHKW % {_KW * _KH}) // {_KW}) - {_PH}, WO * {_SW} + ((CKHKW % {_KW * _KH}) % {_KW}) - {_PW}].when([HO * {_SH} + ((CKHKW % {_KW * _KH}) // {_KW}) - {_PH} >= 0, HO * {_SH} + ((CKHKW % {_KW * _KH}) // {_KW}) - {_PH} < {_H}, WO * {_SW} + ((CKHKW % {_KW * _KH}) % {_KW}) - {_PW} >= 0, WO * {_SW} + ((CKHKW % {_KW * _KH}) % {_KW}) - {_PW} < {_W}], 0.0) * input1[F, CKHKW] where HO in {_HO}, WO in {_WO} \
#             ", { "input0": {"dtype": "float32", "shape": [_N, _CI, _H, _W]}, "input1": {"dtype": "float32", "shape": [_CO, _CI * _KH * _KW]}})' antares

# ConvWinograd_3x3

# COMPUTE_V1='- _N, _CI, _H, _W, _CO = 16, 64, 32, 32, 256; _HO, _WO = _H - 2, _W - 2; _nH, _nW = (_HO + 1) // 2, (_WO + 1) // 2; _P = _N * _nH * _nW; einstein_v2(f"helper4x3[N, M] = const(1.0).when([N * 3 + M == 0, N * 3 + M == 11], const(0.0).when([N * 3 + M == 1, N * 3 + M == 2, N * 3 + M == 9, N * 3 + M == 10], const(-0.5).when([N * 3 + M == 4], 0.5, merge_op=`any`), merge_op=`any`), merge_op=`any`) where N in 4, M in 3; transform_filter[EPS, NU, CI, CO] +=! ((input1[CO, CI, Rkh, Rkw] *
# helper4x3[EPS, Rkh] * helper4x3[NU, Rkw])); input_tile[C, B, EPS, NU] = input0[B // ({_nH} * {_nW}), C, B // {_nW} % {_nH} * 2 + EPS, B % {_nW} * 2 + NU] where C in {_CI}, B in {_P}, EPS in 4, NU in 4; helper4x4[N, M] = const(1.0).when([N * 4 + M == 0, N * 4 + M == 6, N * 4 + M == 9, N * 4 + M == 10, N * 4 + M == 15], const(-1.0).when([N * 4 + M == 5, N * 4 + M == 7, N * 4 + M == 8], 0.0, merge_op=`any`), merge_op=`any`) where N in 4, M in 4; transform_input[EPS, NU, C, B] +=! input_tile[C,
# B, K1, K2] * helper4x4[K1, EPS] * helper4x4[K2, NU] where EPS in 4, NU in 4, C in {_CI}, B in {_P}; batch_gemm[EPS, NU, K, B] +=! transform_filter[EPS, NU, CI, K] * transform_input[EPS, NU, CI, B] where EPS in 4, NU in 4, K in {_CO}, B in {_P}; helper4x2[N, M] = const(0.0) .when([N * 2 + M == 1, N * 2 + M == 6], const(-1.0).when([N * 2 + M == 3], 1.0, merge_op=`any`), merge_op=`any`) where N in 4, M in 2; inverse[K, B, VH, VW] +=! batch_gemm[K1, K2, K, B] * helper4x2[K1, VH] * helper4x2[K2, VW]
# where K in {_CO}, B in {_P}, VH in 2, VW in 2; output0[N, K, H, W] = inverse[K, N * {_nH} * {_nW} + H // 2 * {_nW} + W // 2, H % 2, W % 2] where N in {_N}, K in {_CO}, H in {_HO}, W in {_WO}", {"input0": {"dtype": "float32", "shape": [_N, _CI, _H, _W]}, "input1": {"dtype": "float32", "shape": [_CO, _CI, 3, 3]}})' antares

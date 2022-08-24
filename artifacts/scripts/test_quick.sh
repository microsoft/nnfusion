CHECK=1 BACKEND=c-cuda COMPUTE_V1='- einstein_v2("output0[N, C, H, W] = input0[N, H, W, C]", input_dict={"input0": {"dtype": "float32", "shape": [32, 229, 229, 3]}})' antares

CHECK=1 BACKEND=c-cuda COMPUTE_V1='- einstein_v2("output0[N] = input0[N] + input1[N]", input_dict={"input0": {"dtype": "float32", "shape": [1024 * 512]}, "input1": {"dtype": "float32", "shape": [1024 * 512]}})' antares

CHECK=1 BACKEND=c-cuda COMPUTE_V1='- einstein_v2("output0[A, B, C] = input0[A, B, C / 64, C % 64] where C in 128", input_dict={"input0": {"dtype": "float32", "shape": [3, 3, 2, 64]}})' antares

CHECK=1 BACKEND=c-cuda COMPUTE_V1='- einstein_v2("output0[N, M] = 1.0 / (1.0 + (-input0[N, M]).call(`exp`))", { "input0": {"dtype": "float32", "shape": [1024, 512]} })' antares

CHECK=1 BACKEND=c-cuda COMPUTE_V1='- einstein_v2("output0[N, F] = input0[N, F, 2]", input_dict={"input0": {"dtype": "float32", "shape": [1, 16, 32]}})' antares

CHECK=1 BACKEND=c-cuda COMPUTE_V1='- einstein_v2("output0[N, F, HO, WO] = input0[N] where F in 32, HO in 2, WO in 2", input_dict={"input0": {"dtype": "float32", "shape": [16]}})' antares

CHECK=1 BACKEND=c-cuda COMPUTE_V1='- S = 4096; einstein_v2(input_dict={"input0": {"dtype": "float32", "shape": [S, S]}, "input1": {"dtype": "float32", "shape": [S, S]}}, exprss="output0[N, M] +=! input0[N, K] * input1[K, M]")' antares

CHECK=1 BACKEND=c-cuda COMPUTE_V1='- einstein_v2("output0[N, C, HO, WO] >=! input0[N, C, HO * 2 + KH, WO * 2 + KW] where HO in 6, WO in 6, KW in 2, KH in 2", input_dict={"input0": {"dtype": "float32", "shape": [32, 3, 12, 12]}})' antares

CHECK=1 BACKEND=c-cuda COMPUTE_V1='- einstein_v2("output0[N] +=! input0[N, C]", input_dict={"input0": {"dtype": "float32", "shape": [32, 1024]}})' antares

CHECK=1 BACKEND=c-cuda COMPUTE_V1='- einstein_v2("output0[N, F, HO, WO] +=! input0[N, C, HO + KH, WO + KW] * input1[F, C, KH, KW] where HO in 30, WO in 30", { "input0": {"dtype": "float32", "shape": [16, 64, 32, 32]}, "input1": {"dtype": "float32", "shape": [256, 64, 3, 3]}})' antares

CHECK=1 BACKEND=c-cuda COMPUTE_V1='- einstein_v2("output0[N, C, HO, WO] +=! input0[N, C, HO + KH, WO + KW] * input1[KH, KW, C, 0] where HO in 30, WO in 30", input_dict={"input0": {"dtype": "float32", "shape": [32, 16, 32, 32]}, "input1": {"dtype": "float32", "shape": [3, 3, 16, 1]}})' antares

CHECK=1 BACKEND=c-cuda COMPUTE_V1='- _N, _F, _C, _H, _W, _KH, _KW, _SH, _SW, _PH, _PW = 128, 128, 128, 28, 28, 3, 3, 1, 1, 1, 1; \
              _HO, _WO = (_H - _KH + _PH * 2) // _SH + 1, (_W - _KW + _PW * 2) // _SW + 1; \
              _SA, _RA = _N * _HO * _WO, _C * _KH * _KW; \
              einstein_v2(f" \
                data_pad[RA, SA] = data[SA // {_HO * _WO}, RA // {_KH * _KW}, SA % {_HO * _WO} // {_WO} * {_SH} + RA % {_KH * _KW} // {_KW} - {_PH}, SA % {_HO * _WO} % {_WO} * {_SW} + RA % {_KH * _KW} % {_KW} - {_PW}].when([SA % {_HO * _WO} // {_WO} * {_SH} + RA % {_KH * _KW} // {_KW} >= {_PH}, SA % {_HO * _WO} // {_WO} * {_SH} + RA % {_KH * _KW} // {_KW} < {_H} + {_PH}, SA % {_HO * _WO} % {_WO} * {_SW} + RA % {_KH * _KW} % {_KW} >= {_PW}, SA % {_HO * _WO} % {_WO} * {_SW} + RA % {_KH * _KW} % {_KW} < {_W + _PW}], 0.0) where RA in {_RA}, SA in {_SA}; \
                kernel_pad[F, RA] = kernel[F, RA // {_KH * _KW}, RA % {_KH * _KW} // {_KW}, RA % {_KH * _KW} % {_KW}] where F in {_F}, RA in {_RA}; \
                conv[F, SA] +=! kernel_pad[F, RA] * data_pad[RA, SA] where F in {_F}, SA in {_SA}, RA in {_RA}; \
                conv_unpad[F, SA] = (conv[F, SA] + bias[F]).call(`max`, [0.0]) where F in {_F}, SA in {_SA} \
              ", { "data": {"dtype": "float32", "shape": [_N, _C, _H, _W]}, "kernel": {"dtype": "float32", "shape": [_F, _C, _KH, _KW]}, "bias": {"dtype": "float32", "shape": [_F]}})' antares

CHECK=1 BACKEND=c-cuda COMPUTE_V1='- _N, _C, _H, _W, _KH, _KW, _SH, _SW, _PH, _PW = 128, 84, 83, 83, 5, 5, 2, 2, 2, 2; \
              _HO, _WO = (_H - _KH + _PH * 2) // _SH + 1, (_W - _KW + _PW * 2) // _SW + 1; \
              einstein_v2(f" \
                data_pad[N, C, H, W] = data[N, C, H-{_PH}, W-{_PW}].when([{_PH} <= H, H < {_H+_PH}, {_PW} <= W, W < {_W+_PW}], 0.0) where N in {_N}, C in {_C}, H in {_H + 2 * _PH}, W in {_W + 2 * _PW}; \
                kernel_pad[C, KH, KW] = kernel[C, KH, KW] where C in {_C}, KH in {_KH}, KW in {_KW}; \
                depthwiseconv2d[N, C, H, W] +=! data_pad[N, C, H * {_SH} + KH, W * {_SW} + KW] * kernel_pad[C, KH, KW] where N in {_N}, C in {_C}, H in {_HO}, W in {_WO}, KH in {_KH}, KW in {_KW}; \
                depthwiseconv2d_unpad[N, C, H, W] = depthwiseconv2d[N, C, H, W] where N in {_N}, C in {_C}, H in {_HO}, W in {_WO} \
              ", { "data": {"dtype": "float32", "shape": [_N, _C, _H, _W]}, "kernel": {"dtype": "float32", "shape": [_C, _KH, _KW]}})' antares

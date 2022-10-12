# Depthwise Conv Schedule Fused
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

echo "Finish Depthwise Conv Schedule Fused\n"

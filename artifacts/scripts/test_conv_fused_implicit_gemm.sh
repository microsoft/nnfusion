# Conv Implicit GEMM Fused
# 1. conv S1D1P1 128 128 128 28 28 3 3
# 2. conv S2D1P0 128 128 128 57 57 3 3
# 3. conv S1D1P0 128 168 168 42 42 1 1

CHECK=1 BACKEND=c-cuda COMPUTE_V1='- _N, _F, _C, _H, _W, _KH, _KW, _SH, _SW, _PH, _PW = 128, 128, 128, 28, 28, 3, 3, 1, 1, 1, 1; \
              _HO, _WO = (_H - _KH + _PH * 2) // _SH + 1, (_W - _KW + _PW * 2) // _SW + 1; \
              _SA, _RA = _N * _HO * _WO, _C * _KH * _KW; \
              einstein_v2(f" \
                data_pad[RA, SA] = data[SA // {_HO * _WO}, RA // {_KH * _KW}, SA % {_HO * _WO} // {_WO} * {_SH} + RA % {_KH * _KW} // {_KW} - {_PH}, SA % {_HO * _WO} % {_WO} * {_SW} + RA % {_KH * _KW} % {_KW} - {_PW}].when([SA % {_HO * _WO} // {_WO} * {_SH} + RA % {_KH * _KW} // {_KW} >= {_PH}, SA % {_HO * _WO} // {_WO} * {_SH} + RA % {_KH * _KW} // {_KW} < {_H} + {_PH}, SA % {_HO * _WO} % {_WO} * {_SW} + RA % {_KH * _KW} % {_KW} >= {_PW}, SA % {_HO * _WO} % {_WO} * {_SW} + RA % {_KH * _KW} % {_KW} < {_W + _PW}], 0.0) where RA in {_RA}, SA in {_SA}; \
                kernel_pad[F, RA] = kernel[F, RA // {_KH * _KW}, RA % {_KH * _KW} // {_KW}, RA % {_KH * _KW} % {_KW}] where F in {_F}, RA in {_RA}; \
                conv[F, SA] +=! kernel_pad[F, RA] * data_pad[RA, SA] where F in {_F}, SA in {_SA}, RA in {_RA}; \
                conv_unpad[F, SA] = conv[F, SA] where F in {_F}, SA in {_SA} \
              ", { "data": {"dtype": "float32", "shape": [_N, _C, _H, _W]}, "kernel": {"dtype": "float32", "shape": [_F, _C, _KH, _KW]}})' antares

CHECK=1 BACKEND=c-cuda COMPUTE_V1='- _N, _F, _C, _H, _W, _KH, _KW, _SH, _SW, _PH, _PW = 128, 128, 128, 57, 57, 3, 3, 2, 2, 0, 0; \
              _HO, _WO = (_H - _KH + _PH * 2) // _SH + 1, (_W - _KW + _PW * 2) // _SW + 1; \
              _SA, _RA = _N * _HO * _WO, _C * _KH * _KW; \
              einstein_v2(f" \
                data_pad[RA, SA] = data[SA // {_HO * _WO}, RA // {_KH * _KW}, SA % {_HO * _WO} // {_WO} * {_SH} + RA % {_KH * _KW} // {_KW} - {_PH}, SA % {_HO * _WO} % {_WO} * {_SW} + RA % {_KH * _KW} % {_KW} - {_PW}].when([SA % {_HO * _WO} // {_WO} * {_SH} + RA % {_KH * _KW} // {_KW} >= {_PH}, SA % {_HO * _WO} // {_WO} * {_SH} + RA % {_KH * _KW} // {_KW} < {_H} + {_PH}, SA % {_HO * _WO} % {_WO} * {_SW} + RA % {_KH * _KW} % {_KW} >= {_PW}, SA % {_HO * _WO} % {_WO} * {_SW} + RA % {_KH * _KW} % {_KW} < {_W + _PW}], 0.0) where RA in {_RA}, SA in {_SA}; \
                kernel_pad[F, RA] = kernel[F, RA // {_KH * _KW}, RA % {_KH * _KW} // {_KW}, RA % {_KH * _KW} % {_KW}] where F in {_F}, RA in {_RA}; \
                conv[F, SA] +=! kernel_pad[F, RA] * data_pad[RA, SA] where F in {_F}, SA in {_SA}, RA in {_RA}; \
                conv_unpad[F, SA] = conv[F, SA] where F in {_F}, SA in {_SA} \
              ", { "data": {"dtype": "float32", "shape": [_N, _C, _H, _W]}, "kernel": {"dtype": "float32", "shape": [_F, _C, _KH, _KW]}})' antares

CHECK=1 BACKEND=c-cuda COMPUTE_V1='- _N, _F, _C, _H, _W, _KH, _KW, _SH, _SW, _PH, _PW = 128, 168, 168, 42, 42, 1, 1, 1, 1, 0, 0; \
              _HO, _WO = (_H - _KH + _PH * 2) // _SH + 1, (_W - _KW + _PW * 2) // _SW + 1; \
              _SA, _RA = _N * _HO * _WO, _C * _KH * _KW; \
              einstein_v2(f" \
                data_pad[RA, SA] = data[SA // {_HO * _WO}, RA // {_KH * _KW}, SA % {_HO * _WO} // {_WO} * {_SH} + RA % {_KH * _KW} // {_KW} - {_PH}, SA % {_HO * _WO} % {_WO} * {_SW} + RA % {_KH * _KW} % {_KW} - {_PW}].when([SA % {_HO * _WO} // {_WO} * {_SH} + RA % {_KH * _KW} // {_KW} >= {_PH}, SA % {_HO * _WO} // {_WO} * {_SH} + RA % {_KH * _KW} // {_KW} < {_H} + {_PH}, SA % {_HO * _WO} % {_WO} * {_SW} + RA % {_KH * _KW} % {_KW} >= {_PW}, SA % {_HO * _WO} % {_WO} * {_SW} + RA % {_KH * _KW} % {_KW} < {_W + _PW}], 0.0) where RA in {_RA}, SA in {_SA}; \
                kernel_pad[F, RA] = kernel[F, RA // {_KH * _KW}, RA % {_KH * _KW} // {_KW}, RA % {_KH * _KW} % {_KW}] where F in {_F}, RA in {_RA}; \
                conv[F, SA] +=! kernel_pad[F, RA] * data_pad[RA, SA] where F in {_F}, SA in {_SA}, RA in {_RA}; \
                conv_unpad[F, SA] = conv[F, SA] where F in {_F}, SA in {_SA} \
              ", { "data": {"dtype": "float32", "shape": [_N, _C, _H, _W]}, "kernel": {"dtype": "float32", "shape": [_F, _C, _KH, _KW]}})' antares

echo "Finish Conv Implicit GEMM Fused\n"

# Conv Implicit GEMM Fused Bias
# 1. conv S1D1P1 128 128 128 28 28 3 3
# 2. conv S2D1P0 128 128 128 57 57 3 3
# 3. conv S1D1P0 128 168 168 42 42 1 1

CHECK=1 BACKEND=c-cuda COMPUTE_V1='- _N, _F, _C, _H, _W, _KH, _KW, _SH, _SW, _PH, _PW = 128, 128, 128, 28, 28, 3, 3, 1, 1, 1, 1; \
              _HO, _WO = (_H - _KH + _PH * 2) // _SH + 1, (_W - _KW + _PW * 2) // _SW + 1; \
              _SA, _RA = _N * _HO * _WO, _C * _KH * _KW; \
              einstein_v2(f" \
                data_pad[RA, SA] = data[SA // {_HO * _WO}, RA // {_KH * _KW}, SA % {_HO * _WO} // {_WO} * {_SH} + RA % {_KH * _KW} // {_KW} - {_PH}, SA % {_HO * _WO} % {_WO} * {_SW} + RA % {_KH * _KW} % {_KW} - {_PW}].when([SA % {_HO * _WO} // {_WO} * {_SH} + RA % {_KH * _KW} // {_KW} >= {_PH}, SA % {_HO * _WO} // {_WO} * {_SH} + RA % {_KH * _KW} // {_KW} < {_H} + {_PH}, SA % {_HO * _WO} % {_WO} * {_SW} + RA % {_KH * _KW} % {_KW} >= {_PW}, SA % {_HO * _WO} % {_WO} * {_SW} + RA % {_KH * _KW} % {_KW} < {_W + _PW}], 0.0) where RA in {_RA}, SA in {_SA}; \
                kernel_pad[F, RA] = kernel[F, RA // {_KH * _KW}, RA % {_KH * _KW} // {_KW}, RA % {_KH * _KW} % {_KW}] where F in {_F}, RA in {_RA}; \
                conv[F, SA] +=! kernel_pad[F, RA] * data_pad[RA, SA] where F in {_F}, SA in {_SA}, RA in {_RA}; \
                conv_unpad[F, SA] = conv[F, SA] + bias[F] where F in {_F}, SA in {_SA} \
              ", { "data": {"dtype": "float32", "shape": [_N, _C, _H, _W]}, "kernel": {"dtype": "float32", "shape": [_F, _C, _KH, _KW]}, "bias": {"dtype": "float32", "shape": [_F]}})' antares

CHECK=1 BACKEND=c-cuda COMPUTE_V1='- _N, _F, _C, _H, _W, _KH, _KW, _SH, _SW, _PH, _PW = 128, 128, 128, 57, 57, 3, 3, 2, 2, 0, 0; \
              _HO, _WO = (_H - _KH + _PH * 2) // _SH + 1, (_W - _KW + _PW * 2) // _SW + 1; \
              _SA, _RA = _N * _HO * _WO, _C * _KH * _KW; \
              einstein_v2(f" \
                data_pad[RA, SA] = data[SA // {_HO * _WO}, RA // {_KH * _KW}, SA % {_HO * _WO} // {_WO} * {_SH} + RA % {_KH * _KW} // {_KW} - {_PH}, SA % {_HO * _WO} % {_WO} * {_SW} + RA % {_KH * _KW} % {_KW} - {_PW}].when([SA % {_HO * _WO} // {_WO} * {_SH} + RA % {_KH * _KW} // {_KW} >= {_PH}, SA % {_HO * _WO} // {_WO} * {_SH} + RA % {_KH * _KW} // {_KW} < {_H} + {_PH}, SA % {_HO * _WO} % {_WO} * {_SW} + RA % {_KH * _KW} % {_KW} >= {_PW}, SA % {_HO * _WO} % {_WO} * {_SW} + RA % {_KH * _KW} % {_KW} < {_W + _PW}], 0.0) where RA in {_RA}, SA in {_SA}; \
                kernel_pad[F, RA] = kernel[F, RA // {_KH * _KW}, RA % {_KH * _KW} // {_KW}, RA % {_KH * _KW} % {_KW}] where F in {_F}, RA in {_RA}; \
                conv[F, SA] +=! kernel_pad[F, RA] * data_pad[RA, SA] where F in {_F}, SA in {_SA}, RA in {_RA}; \
                conv_unpad[F, SA] = conv[F, SA] + bias[F] where F in {_F}, SA in {_SA} \
              ", { "data": {"dtype": "float32", "shape": [_N, _C, _H, _W]}, "kernel": {"dtype": "float32", "shape": [_F, _C, _KH, _KW]}, "bias": {"dtype": "float32", "shape": [_F]}})' antares

CHECK=1 BACKEND=c-cuda COMPUTE_V1='- _N, _F, _C, _H, _W, _KH, _KW, _SH, _SW, _PH, _PW = 128, 168, 168, 42, 42, 1, 1, 1, 1, 0, 0; \
              _HO, _WO = (_H - _KH + _PH * 2) // _SH + 1, (_W - _KW + _PW * 2) // _SW + 1; \
              _SA, _RA = _N * _HO * _WO, _C * _KH * _KW; \
              einstein_v2(f" \
                data_pad[RA, SA] = data[SA // {_HO * _WO}, RA // {_KH * _KW}, SA % {_HO * _WO} // {_WO} * {_SH} + RA % {_KH * _KW} // {_KW} - {_PH}, SA % {_HO * _WO} % {_WO} * {_SW} + RA % {_KH * _KW} % {_KW} - {_PW}].when([SA % {_HO * _WO} // {_WO} * {_SH} + RA % {_KH * _KW} // {_KW} >= {_PH}, SA % {_HO * _WO} // {_WO} * {_SH} + RA % {_KH * _KW} // {_KW} < {_H} + {_PH}, SA % {_HO * _WO} % {_WO} * {_SW} + RA % {_KH * _KW} % {_KW} >= {_PW}, SA % {_HO * _WO} % {_WO} * {_SW} + RA % {_KH * _KW} % {_KW} < {_W + _PW}], 0.0) where RA in {_RA}, SA in {_SA}; \
                kernel_pad[F, RA] = kernel[F, RA // {_KH * _KW}, RA % {_KH * _KW} // {_KW}, RA % {_KH * _KW} % {_KW}] where F in {_F}, RA in {_RA}; \
                conv[F, SA] +=! kernel_pad[F, RA] * data_pad[RA, SA] where F in {_F}, SA in {_SA}, RA in {_RA}; \
                conv_unpad[F, SA] = conv[F, SA] + bias[F] where F in {_F}, SA in {_SA} \
              ", { "data": {"dtype": "float32", "shape": [_N, _C, _H, _W]}, "kernel": {"dtype": "float32", "shape": [_F, _C, _KH, _KW]}, "bias": {"dtype": "float32", "shape": [_F]}})' antares

echo "Finish Conv Implicit GEMM Fused Bias"

# Conv Implicit GEMM Fused Relu
# 1. conv S1D1P1 128 128 128 28 28 3 3
# 2. conv S2D1P0 128 128 128 57 57 3 3
# 3. conv S1D1P0 128 168 168 42 42 1 1

CHECK=1 BACKEND=c-cuda COMPUTE_V1='- _N, _F, _C, _H, _W, _KH, _KW, _SH, _SW, _PH, _PW = 128, 128, 128, 28, 28, 3, 3, 1, 1, 1, 1; \
              _HO, _WO = (_H - _KH + _PH * 2) // _SH + 1, (_W - _KW + _PW * 2) // _SW + 1; \
              _SA, _RA = _N * _HO * _WO, _C * _KH * _KW; \
              einstein_v2(f" \
                data_pad[RA, SA] = data[SA // {_HO * _WO}, RA // {_KH * _KW}, SA % {_HO * _WO} // {_WO} * {_SH} + RA % {_KH * _KW} // {_KW} - {_PH}, SA % {_HO * _WO} % {_WO} * {_SW} + RA % {_KH * _KW} % {_KW} - {_PW}].when([SA % {_HO * _WO} // {_WO} * {_SH} + RA % {_KH * _KW} // {_KW} >= {_PH}, SA % {_HO * _WO} // {_WO} * {_SH} + RA % {_KH * _KW} // {_KW} < {_H} + {_PH}, SA % {_HO * _WO} % {_WO} * {_SW} + RA % {_KH * _KW} % {_KW} >= {_PW}, SA % {_HO * _WO} % {_WO} * {_SW} + RA % {_KH * _KW} % {_KW} < {_W + _PW}], 0.0) where RA in {_RA}, SA in {_SA}; \
                kernel_pad[F, RA] = kernel[F, RA // {_KH * _KW}, RA % {_KH * _KW} // {_KW}, RA % {_KH * _KW} % {_KW}] where F in {_F}, RA in {_RA}; \
                conv[F, SA] +=! kernel_pad[F, RA] * data_pad[RA, SA] where F in {_F}, SA in {_SA}, RA in {_RA}; \
                conv_unpad[F, SA] = conv[F, SA].call(`max`, [0.0]) where F in {_F}, SA in {_SA} \
              ", { "data": {"dtype": "float32", "shape": [_N, _C, _H, _W]}, "kernel": {"dtype": "float32", "shape": [_F, _C, _KH, _KW]}})' antares

CHECK=1 BACKEND=c-cuda COMPUTE_V1='- _N, _F, _C, _H, _W, _KH, _KW, _SH, _SW, _PH, _PW = 128, 128, 128, 57, 57, 3, 3, 2, 2, 0, 0; \
              _HO, _WO = (_H - _KH + _PH * 2) // _SH + 1, (_W - _KW + _PW * 2) // _SW + 1; \
              _SA, _RA = _N * _HO * _WO, _C * _KH * _KW; \
              einstein_v2(f" \
                data_pad[RA, SA] = data[SA // {_HO * _WO}, RA // {_KH * _KW}, SA % {_HO * _WO} // {_WO} * {_SH} + RA % {_KH * _KW} // {_KW} - {_PH}, SA % {_HO * _WO} % {_WO} * {_SW} + RA % {_KH * _KW} % {_KW} - {_PW}].when([SA % {_HO * _WO} // {_WO} * {_SH} + RA % {_KH * _KW} // {_KW} >= {_PH}, SA % {_HO * _WO} // {_WO} * {_SH} + RA % {_KH * _KW} // {_KW} < {_H} + {_PH}, SA % {_HO * _WO} % {_WO} * {_SW} + RA % {_KH * _KW} % {_KW} >= {_PW}, SA % {_HO * _WO} % {_WO} * {_SW} + RA % {_KH * _KW} % {_KW} < {_W + _PW}], 0.0) where RA in {_RA}, SA in {_SA}; \
                kernel_pad[F, RA] = kernel[F, RA // {_KH * _KW}, RA % {_KH * _KW} // {_KW}, RA % {_KH * _KW} % {_KW}] where F in {_F}, RA in {_RA}; \
                conv[F, SA] +=! kernel_pad[F, RA] * data_pad[RA, SA] where F in {_F}, SA in {_SA}, RA in {_RA}; \
                conv_unpad[F, SA] = conv[F, SA].call(`max`, [0.0]) where F in {_F}, SA in {_SA} \
              ", { "data": {"dtype": "float32", "shape": [_N, _C, _H, _W]}, "kernel": {"dtype": "float32", "shape": [_F, _C, _KH, _KW]}})' antares

CHECK=1 BACKEND=c-cuda COMPUTE_V1='- _N, _F, _C, _H, _W, _KH, _KW, _SH, _SW, _PH, _PW = 128, 168, 168, 42, 42, 1, 1, 1, 1, 0, 0; \
              _HO, _WO = (_H - _KH + _PH * 2) // _SH + 1, (_W - _KW + _PW * 2) // _SW + 1; \
              _SA, _RA = _N * _HO * _WO, _C * _KH * _KW; \
              einstein_v2(f" \
                data_pad[RA, SA] = data[SA // {_HO * _WO}, RA // {_KH * _KW}, SA % {_HO * _WO} // {_WO} * {_SH} + RA % {_KH * _KW} // {_KW} - {_PH}, SA % {_HO * _WO} % {_WO} * {_SW} + RA % {_KH * _KW} % {_KW} - {_PW}].when([SA % {_HO * _WO} // {_WO} * {_SH} + RA % {_KH * _KW} // {_KW} >= {_PH}, SA % {_HO * _WO} // {_WO} * {_SH} + RA % {_KH * _KW} // {_KW} < {_H} + {_PH}, SA % {_HO * _WO} % {_WO} * {_SW} + RA % {_KH * _KW} % {_KW} >= {_PW}, SA % {_HO * _WO} % {_WO} * {_SW} + RA % {_KH * _KW} % {_KW} < {_W + _PW}], 0.0) where RA in {_RA}, SA in {_SA}; \
                kernel_pad[F, RA] = kernel[F, RA // {_KH * _KW}, RA % {_KH * _KW} // {_KW}, RA % {_KH * _KW} % {_KW}] where F in {_F}, RA in {_RA}; \
                conv[F, SA] +=! kernel_pad[F, RA] * data_pad[RA, SA] where F in {_F}, SA in {_SA}, RA in {_RA}; \
                conv_unpad[F, SA] = conv[F, SA].call(`max`, [0.0]) where F in {_F}, SA in {_SA} \
              ", { "data": {"dtype": "float32", "shape": [_N, _C, _H, _W]}, "kernel": {"dtype": "float32", "shape": [_F, _C, _KH, _KW]}})' antares

echo "Finish Conv Implicit GEMM Fused Relu"

# Conv Implicit GEMM Fused Bias Relu
# 1. conv S1D1P1 128 128 128 28 28 3 3
# 2. conv S2D1P0 128 128 128 57 57 3 3
# 3. conv S1D1P0 128 168 168 42 42 1 1

CHECK=1 BACKEND=c-cuda COMPUTE_V1='- _N, _F, _C, _H, _W, _KH, _KW, _SH, _SW, _PH, _PW = 128, 128, 128, 28, 28, 3, 3, 1, 1, 1, 1; \
              _HO, _WO = (_H - _KH + _PH * 2) // _SH + 1, (_W - _KW + _PW * 2) // _SW + 1; \
              _SA, _RA = _N * _HO * _WO, _C * _KH * _KW; \
              einstein_v2(f" \
                data_pad[RA, SA] = data[SA // {_HO * _WO}, RA // {_KH * _KW}, SA % {_HO * _WO} // {_WO} * {_SH} + RA % {_KH * _KW} // {_KW} - {_PH}, SA % {_HO * _WO} % {_WO} * {_SW} + RA % {_KH * _KW} % {_KW} - {_PW}].when([SA % {_HO * _WO} // {_WO} * {_SH} + RA % {_KH * _KW} // {_KW} >= {_PH}, SA % {_HO * _WO} // {_WO} * {_SH} + RA % {_KH * _KW} // {_KW} < {_H} + {_PH}, SA % {_HO * _WO} % {_WO} * {_SW} + RA % {_KH * _KW} % {_KW} >= {_PW}, SA % {_HO * _WO} % {_WO} * {_SW} + RA % {_KH * _KW} % {_KW} < {_W + _PW}], 0.0) where RA in {_RA}, SA in {_SA}; \
                kernel_pad[F, RA] = kernel[F, RA // {_KH * _KW}, RA % {_KH * _KW} // {_KW}, RA % {_KH * _KW} % {_KW}] where F in {_F}, RA in {_RA}; \
                conv[F, SA] +=! kernel_pad[F, RA] * data_pad[RA, SA] where F in {_F}, SA in {_SA}, RA in {_RA}; \
                conv_unpad[F, SA] = (conv[F, SA] + bias[F]).call(`max`, [0.0]) where F in {_F}, SA in {_SA} \
              ", { "data": {"dtype": "float32", "shape": [_N, _C, _H, _W]}, "kernel": {"dtype": "float32", "shape": [_F, _C, _KH, _KW]}, "bias": {"dtype": "float32", "shape": [_F]}})' antares

CHECK=1 BACKEND=c-cuda COMPUTE_V1='- _N, _F, _C, _H, _W, _KH, _KW, _SH, _SW, _PH, _PW = 128, 128, 128, 57, 57, 3, 3, 2, 2, 0, 0; \
              _HO, _WO = (_H - _KH + _PH * 2) // _SH + 1, (_W - _KW + _PW * 2) // _SW + 1; \
              _SA, _RA = _N * _HO * _WO, _C * _KH * _KW; \
              einstein_v2(f" \
                data_pad[RA, SA] = data[SA // {_HO * _WO}, RA // {_KH * _KW}, SA % {_HO * _WO} // {_WO} * {_SH} + RA % {_KH * _KW} // {_KW} - {_PH}, SA % {_HO * _WO} % {_WO} * {_SW} + RA % {_KH * _KW} % {_KW} - {_PW}].when([SA % {_HO * _WO} // {_WO} * {_SH} + RA % {_KH * _KW} // {_KW} >= {_PH}, SA % {_HO * _WO} // {_WO} * {_SH} + RA % {_KH * _KW} // {_KW} < {_H} + {_PH}, SA % {_HO * _WO} % {_WO} * {_SW} + RA % {_KH * _KW} % {_KW} >= {_PW}, SA % {_HO * _WO} % {_WO} * {_SW} + RA % {_KH * _KW} % {_KW} < {_W + _PW}], 0.0) where RA in {_RA}, SA in {_SA}; \
                kernel_pad[F, RA] = kernel[F, RA // {_KH * _KW}, RA % {_KH * _KW} // {_KW}, RA % {_KH * _KW} % {_KW}] where F in {_F}, RA in {_RA}; \
                conv[F, SA] +=! kernel_pad[F, RA] * data_pad[RA, SA] where F in {_F}, SA in {_SA}, RA in {_RA}; \
                conv_unpad[F, SA] = (conv[F, SA] + bias[F]).call(`max`, [0.0]) where F in {_F}, SA in {_SA} \
              ", { "data": {"dtype": "float32", "shape": [_N, _C, _H, _W]}, "kernel": {"dtype": "float32", "shape": [_F, _C, _KH, _KW]}, "bias": {"dtype": "float32", "shape": [_F]}})' antares

CHECK=1 BACKEND=c-cuda COMPUTE_V1='- _N, _F, _C, _H, _W, _KH, _KW, _SH, _SW, _PH, _PW = 128, 168, 168, 42, 42, 1, 1, 1, 1, 0, 0; \
              _HO, _WO = (_H - _KH + _PH * 2) // _SH + 1, (_W - _KW + _PW * 2) // _SW + 1; \
              _SA, _RA = _N * _HO * _WO, _C * _KH * _KW; \
              einstein_v2(f" \
                data_pad[RA, SA] = data[SA // {_HO * _WO}, RA // {_KH * _KW}, SA % {_HO * _WO} // {_WO} * {_SH} + RA % {_KH * _KW} // {_KW} - {_PH}, SA % {_HO * _WO} % {_WO} * {_SW} + RA % {_KH * _KW} % {_KW} - {_PW}].when([SA % {_HO * _WO} // {_WO} * {_SH} + RA % {_KH * _KW} // {_KW} >= {_PH}, SA % {_HO * _WO} // {_WO} * {_SH} + RA % {_KH * _KW} // {_KW} < {_H} + {_PH}, SA % {_HO * _WO} % {_WO} * {_SW} + RA % {_KH * _KW} % {_KW} >= {_PW}, SA % {_HO * _WO} % {_WO} * {_SW} + RA % {_KH * _KW} % {_KW} < {_W + _PW}], 0.0) where RA in {_RA}, SA in {_SA}; \
                kernel_pad[F, RA] = kernel[F, RA // {_KH * _KW}, RA % {_KH * _KW} // {_KW}, RA % {_KH * _KW} % {_KW}] where F in {_F}, RA in {_RA}; \
                conv[F, SA] +=! kernel_pad[F, RA] * data_pad[RA, SA] where F in {_F}, SA in {_SA}, RA in {_RA}; \
                conv_unpad[F, SA] = (conv[F, SA] + bias[F]).call(`max`, [0.0]) where F in {_F}, SA in {_SA} \
              ", { "data": {"dtype": "float32", "shape": [_N, _C, _H, _W]}, "kernel": {"dtype": "float32", "shape": [_F, _C, _KH, _KW]}, "bias": {"dtype": "float32", "shape": [_F]}})' antares

echo "Finish Conv Implicit GEMM Fused Bias Relu"

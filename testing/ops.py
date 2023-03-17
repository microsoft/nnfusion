def matmul_nn(n, k, m, dtype="float32"):
    ir = "output0[N, M] +=! input0[N, K] * input1[K, M]"
    input_dict = {
        "input0": {"dtype": dtype, "shape": [n, k]},
        "input1": {"dtype": dtype, "shape": [k, m]}
    }
    return ir, input_dict

def matmul_nn_bias(n, k, m, dtype="float32"):
    ir = "mediate0[N, M] +=! input0[N, K] * input1[K, M]; output0[N, M] = mediate0[N, M] + input2[M];"
    input_dict = {
        "input0": {"dtype": dtype, "shape": [n, k]},
        "input1": {"dtype": dtype, "shape": [k, m]},
        "input2": {"dtype": dtype, "shape": [m]},
    }
    return ir, input_dict

def matmul_nt(n, k, m, dtype="float32"):
    ir = "output0[N, M] +=! input0[N, K] * input1[M, K]"
    input_dict = {
        "input0": {"dtype": dtype, "shape": [n, k]},
        "input1": {"dtype": dtype, "shape": [m, k]}
    }
    return ir, input_dict

def matmul_tn(n, k, m, dtype="float32"):
    ir = "output0[N, M] +=! input0[K, N] * input1[K, M]"
    input_dict = {
        "input0": {"dtype": dtype, "shape": [k, n]},
        "input1": {"dtype": dtype, "shape": [k, m]}
    }
    return ir, input_dict

def conv_nchw(n, f, h, w, c, kh, kw, s, d, p, dtype="float32"):
    if p == 0:
        return conv_nchw_nopad(n, f, h, w, c, kh, kw, s, d, p, dtype)
    inh = (h - 1) * s + (kh - 1) * d + 1 - 2 * p
    inw = (w - 1) * s + (kw - 1) * d + 1 - 2 * p
    padh = inh + 2 * p
    padw = inw + 2 * p
    ir = f"pad[N, C, H0, W0] = input0[N, C, H0-{p}, W0-{p}].when([H0>={p}, H0<{inh+p}, W0>={p}, W0<{inw+p}], const(0.0).cast(input0[N, C, H0-{p}, W0-{p}].dtype())) where H0 in {padh}, W0 in {padw}; \
           output0[N, F, HO, WO] +=! pad[N, C, KH*{d} + HO*{s}, KW*{d} + WO*{s}] * input1[F, C, KH, KW] where HO in {h}, WO in {w};"
    input_dict = {
        "input0": {"dtype": dtype, "shape": [n, c, inh, inw]},
        "input1": {"dtype": dtype, "shape": [f, c, kh, kw]}
    }
    return ir, input_dict

def conv_nchw_nopad(n, f, h, w, c, kh, kw, s, d, p, dtype="float32"):
    assert p == 0
    inh = (h - 1) * s + (kh - 1) * d + 1 - 2 * p
    inw = (w - 1) * s + (kw - 1) * d + 1 - 2 * p
    ir = f"output0[N, F, HO, WO] +=! input0[N, C, KH*{d} + HO*{s}, KW*{d} + WO*{s}] * input1[F, C, KH, KW] where HO in {h}, WO in {w};"
    input_dict = {
        "input0": {"dtype": dtype, "shape": [n, c, inh, inw]},
        "input1": {"dtype": dtype, "shape": [f, c, kh, kw]}
    }
    return ir, input_dict

def relu(n, dtype="float32"):
    ir = "output0[N0] = input0[N0].call(`max`, [const(0).cast(input0[N0].dtype())])"
    input_dict = {
        "input0": {"dtype": dtype, "shape": [n]}
    }
    return ir, input_dict

def row_reduce(n, m, dtype="float32"):
    ir = "output0[N] +=! input0[N, K]"
    input_dict = {
        "input0": {"dtype": dtype, "shape": [n, m]},
    }
    return ir, input_dict

def dwconv_nchw(n, c, h, w, kh, kw, s, d, p, dtype="float32"):
    inh = (h - 1) * s + (kh - 1) * d + 1 - 2 * p
    inw = (w - 1) * s + (kw - 1) * d + 1 - 2 * p
    padh = inh + 2 * p
    padw = inw + 2 * p
    ir = f"pad[N, C, H0, W0] = input0[N, C, H0-{p}, W0-{p}].when([H0>={p}, H0<{inh+p}, W0>={p}, W0<{inw+p}], const(0.0).cast(input0[N, C, H0-{p}, W0-{p}].dtype())) where H0 in {padh}, W0 in {padw}; \
           output0[N, C, HO, WO] +=! pad[N, C, KH*{d} + HO*{s}, KW*{d} + WO*{s}] * input1[C, KH, KW] where HO in {h}, WO in {w};"
    input_dict = {
        "input0": {"dtype": dtype, "shape": [n, c, inh, inw]},
        "input1": {"dtype": dtype, "shape": [c, kh, kw]}
    }
    return ir, input_dict

def average_pooling(b, h, w, kh, kw, s, p, dtype="float32"):
    inh = (h - 1) * s + kh - 2 * p
    inw = (w - 1) * s + kw - 2 * p
    padh = inh + 2 * p
    padw = inw + 2 * p
    ir = f"pad[B, H0, W0] = input0[B, H0-{p}, W0-{p}].when([H0>={p}, H0<{inh+p}, W0>={p}, W0<{inw+p}], const(0.0).cast(input0[B, H0-{p}, W0-{p}].dtype())) where H0 in {padh}, W0 in {padw}; \
           output0[B, HO, WO] +=! pad[B, KH + HO*{s}, KW + WO*{s}] where HO in {h}, WO in {w}, KH in {kh}, KW in {kw};"
    input_dict = {
        "input0": {"dtype": dtype, "shape": [b, inh, inw]},
    }
    return ir, input_dict

def transpose(n, m, dtype="float32"):
    ir = "output0[N, M] = input0[M, N]"
    input_dict = {
        "input0": {"dtype": dtype, "shape": [n, m]},
    }
    return ir, input_dict

def conv_nhwc(n, f, h, w, c, kh, kw, s, d, p, dtype="float32"):
    if p == 0:
        return conv_nhwc_nopad(n, f, h, w, c, kh, kw, s, d, p, dtype)
    inh = (h - 1) * s + (kh - 1) * d + 1 - 2 * p
    inw = (w - 1) * s + (kw - 1) * d + 1 - 2 * p
    padh = inh + 2 * p
    padw = inw + 2 * p
    ir = f"pad[N, H0, W0, C] = input0[N, H0-{p}, W0-{p}, C].when([H0>={p}, H0<{inh+p}, W0>={p}, W0<{inw+p}], const(0.0).cast(input0[N, H0-{p}, W0-{p}, C].dtype())) where H0 in {padh}, W0 in {padw}; \
           output0[N, HO, WO, F] +=! pad[N, KH*{d} + HO*{s}, KW*{d} + WO*{s}, C] * input1[KH, KW, C, F] where HO in {h}, WO in {w};"
    input_dict = {
        "input0": {"dtype": dtype, "shape": [n, inh, inw, c]},
        "input1": {"dtype": dtype, "shape": [kh, kw, c, f]}
    }
    return ir, input_dict

def conv_nhwc_nopad(n, f, h, w, c, kh, kw, s, d, p, dtype="float32"):
    assert p == 0
    inh = (h - 1) * s + (kh - 1) * d + 1 - 2 * p
    inw = (w - 1) * s + (kw - 1) * d + 1 - 2 * p
    ir = f"output0[N, HO, WO, F] +=! input0[N, KH*{d} + HO*{s}, KW*{d} + WO*{s}, C] * input1[KH, KW, C, F] where HO in {h}, WO in {w};"
    input_dict = {
        "input0": {"dtype": dtype, "shape": [n, inh, inw, c]},
        "input1": {"dtype": dtype, "shape": [kh, kw, c, f]}
    }
    return ir, input_dict

def dwconv_nhwc(n, c, h, w, kh, kw, s, d, p, dtype="float32"):
    inh = (h - 1) * s + (kh - 1) * d + 1 - 2 * p
    inw = (w - 1) * s + (kw - 1) * d + 1 - 2 * p
    padh = inh + 2 * p
    padw = inw + 2 * p
    ir = f"pad[N, H0, W0, C] = input0[N, H0-{p}, W0-{p}, C].when([H0>={p}, H0<{inh+p}, W0>={p}, W0<{inw+p}], const(0.0).cast(input0[N, H0-{p}, W0-{p}, C].dtype())) where H0 in {padh}, W0 in {padw}; \
           output0[N, HO, WO, C] +=! pad[N, KH*{d} + HO*{s}, KW*{d} + WO*{s}, C] * input1[KH, KW, C] where HO in {h}, WO in {w};"
    input_dict = {
        "input0": {"dtype": dtype, "shape": [n, inh, inw, c]},
        "input1": {"dtype": dtype, "shape": [kh, kw, c]}
    }
    return ir, input_dict

def conv_nchw_implicit_gemm(n, f, h, w, c, kh, kw, s, d, p, dtype="float16"):
    if p == 0:
        return conv_nchw_implicit_gemm_no_pad(n, f, h, w, c, kh, kw, s, d, p, dtype)
    inh = (h - 1) * s + (kh - 1) * d + 1 - 2 * p
    inw = (w - 1) * s + (kw - 1) * d + 1 - 2 * p
    padh = inh + 2 * p
    padw = inw + 2 * p
    inh = (h - 1) * s + (kh - 1) * d + 1 - 2 * p
    inw = (w - 1) * s + (kw - 1) * d + 1 - 2 * p
    ir = f"pad[N, C, H0, W0] = input0[N, C, H0-{p}, W0-{p}].when([H0>={p}, H0<{inh+p}, W0>={p}, W0<{inw+p}], const(0.0).cast(input0[N, C, H0-{p}, W0-{p}].dtype())) where H0 in {padh}, W0 in {padw}; \
           data[K, N] = pad[N//{h*w}, K//{kh*kw}, N%{h*w}//{w}*{s}+K%{kh*kw}//{kw}*{d}, N%{w}*{s}+K%{kw}*{d}] where K in {kh*kw*c}, N in {n*h*w}; \
           output0[M, N] +=! input1[M, K] * data[K, N];"
    input_dict = {
        "input0": {"dtype": dtype, "shape": [n, c, inh, inw]},
        "input1": {"dtype": dtype, "shape": [f, c * kh * kw]}
    }
    return ir, input_dict


def conv_nchw_implicit_gemm_no_pad(n, f, h, w, c, kh, kw, s, d, p, dtype="float16"):
    assert p == 0
    inh = (h - 1) * s + (kh - 1) * d + 1 - 2 * p
    inw = (w - 1) * s + (kw - 1) * d + 1 - 2 * p
    ir = f"data[K, N] = input0[N//{h*w}, K//{kh*kw}, N%{h*w}//{w}*{s}+K%{kh*kw}//{kw}*{d}, N%{w}*{s}+K%{kw}*{d}] where K in {kh*kw*c}, N in {n*h*w}; \
           output0[M, N] +=! input1[M, K] * data[K, N];"
    input_dict = {
        "input0": {"dtype": dtype, "shape": [n, c, inh, inw]},
        "input1": {"dtype": dtype, "shape": [f, c * kh * kw]}
    }
    return ir, input_dict

def conv_nhwc_implicit_gemm(n, f, h, w, c, kh, kw, s, d, p, dtype="float16"):
    if p == 0:
        return conv_nhwc_implicit_gemm_no_pad(n, f, h, w, c, kh, kw, s, d, p)
    inh = (h - 1) * s + (kh - 1) * d + 1 - 2 * p
    inw = (w - 1) * s + (kw - 1) * d + 1 - 2 * p
    padh = inh + 2 * p
    padw = inw + 2 * p
    ir = f"pad[N, H0, W0, C] = input0[N, H0-{p}, W0-{p}, C].when([H0>={p}, H0<{inh+p}, W0>={p}, W0<{inw+p}], const(0.0).cast(input0[N, C, H0-{p}, W0-{p}].dtype())) where H0 in {padh}, W0 in {padw}; \
           data[N, K] = pad[N//{h*w}, N%{h*w}//{w}*{s}+K//{kw*c}*{d}, N%{w}*{s}+K//{c}%{kw}*{d}, K%{c}] where K in {kh*kw*c}, N in {n*h*w}; \
           output0[N, M] +=! data[N, K] * input1[K, M];"
    input_dict = {
        "input0": {"dtype": dtype, "shape": [n, inh, inw, c]},
        "input1": {"dtype": dtype, "shape": [kh*kw*c, f]}
    }
    return ir, input_dict

def conv_nhwc_implicit_gemm_no_pad(n, f, h, w, c, kh, kw, s, d, p, dtype="float16"):
    assert p == 0
    inh = (h - 1) * s + (kh - 1) * d + 1 - 2 * p
    inw = (w - 1) * s + (kw - 1) * d + 1 - 2 * p
    ir = f"data[N, K] = input0[N//{h*w}, N%{h*w}//{w}*{s}+K//{kw*c}*{d}, N%{w}*{s}+K//{c}%{kw}*{d}, K%{c}] where K in {kh*kw*c}, N in {n*h*w}; \
           output0[N, M] +=! data[N, K] * input1[K, M];"
    input_dict = {
        "input0": {"dtype": dtype, "shape": [n, inh, inw, c]},
        "input1": {"dtype": dtype, "shape": [kh*kw*c, f]}
    }
    return ir, input_dict

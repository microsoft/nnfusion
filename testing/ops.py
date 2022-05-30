def matmul_nn(n, k, m):
    ir = "output0[N, M] +=! input0[N, K] * input1[K, M]"
    input_dict = {
        "input0": {"dtype": "float32", "shape": [n, k]},
        "input1": {"dtype": "float32", "shape": [k, m]}
    }
    return ir, input_dict

def matmul_nt(n, k, m):
    ir = "output0[N, M] +=! input0[N, K] * input1[M, K]"
    input_dict = {
        "input0": {"dtype": "float32", "shape": [n, k]},
        "input1": {"dtype": "float32", "shape": [m, k]}
    }
    return ir, input_dict

def matmul_tn(n, k, m):
    ir = "output0[N, M] +=! input0[K, N] * input1[K, M]"
    input_dict = {
        "input0": {"dtype": "float32", "shape": [k, n]},
        "input1": {"dtype": "float32", "shape": [k, m]}
    }
    return ir, input_dict

def conv_nchw(n, f, h, w, c, kh, kw, s, d, p):
    if p == 0:
        return conv_nchw_nopad(n, f, h, w, c, kh, kw, s, d, p)
    inh = (h - 1) * s + (kh - 1) * d + 1 - 2 * p
    inw = (w - 1) * s + (kw - 1) * d + 1 - 2 * p
    padh = inh + 2 * p
    padw = inw + 2 * p
    ir = f"pad[N, C, H0, W0] = input0[N, C, H0-{p}, W0-{p}].when([H0>={p}, H0<{inh+p}, W0>={p}, W0<{inw+p}], const(0.0).cast(input0[N, C, H0-{p}, W0-{p}].dtype())) where H0 in {padh}, W0 in {padw}; \
           output0[N, F, HO, WO] +=! pad[N, C, KH*{d} + HO*{s}, KW*{d} + WO*{s}] * input1[F, C, KH, KW] where HO in {h}, WO in {w};"
    input_dict = {
        "input0": {"dtype": "float32", "shape": [n, c, inh, inw]},
        "input1": {"dtype": "float32", "shape": [f, c, kh, kw]}
    }
    return ir, input_dict

def conv_nchw_nopad(n, f, h, w, c, kh, kw, s, d, p):
    assert p == 0
    inh = (h - 1) * s + (kh - 1) * d + 1 - 2 * p
    inw = (w - 1) * s + (kw - 1) * d + 1 - 2 * p
    ir = f"output0[N, F, HO, WO] +=! input0[N, C, KH*{d} + HO*{s}, KW*{d} + WO*{s}] * input1[F, C, KH, KW] where HO in {h}, WO in {w};"
    input_dict = {
        "input0": {"dtype": "float32", "shape": [n, c, inh, inw]},
        "input1": {"dtype": "float32", "shape": [f, c, kh, kw]}
    }
    return ir, input_dict

def relu(n):
    ir = "output0[N0] = input0[N0].call(`max`, [const(0).cast(input0[N0].dtype())])"
    input_dict = {
        "input0": {"dtype": "float32", "shape": [n]}
    }
    return ir, input_dict

def row_reduce(n, m):
    ir = "output0[N] +=! input0[N, K]"
    input_dict = {
        "input0": {"dtype": "float32", "shape": [n, m]},
    }
    return ir, input_dict

def dwconv_nchw(n, c, h, w, kh, kw, s, d, p):
    inh = (h - 1) * s + (kh - 1) * d + 1 - 2 * p
    inw = (w - 1) * s + (kw - 1) * d + 1 - 2 * p
    padh = inh + 2 * p
    padw = inw + 2 * p
    ir = f"pad[N, C, H0, W0] = input0[N, C, H0-{p}, W0-{p}].when([H0>={p}, H0<{inh+p}, W0>={p}, W0<{inw+p}], const(0.0).cast(input0[N, C, H0-{p}, W0-{p}].dtype())) where H0 in {padh}, W0 in {padw}; \
           output0[N, C, HO, WO] +=! pad[N, C, KH*{d} + HO*{s}, KW*{d} + WO*{s}] * input1[C, KH, KW] where HO in {h}, WO in {w};"
    input_dict = {
        "input0": {"dtype": "float32", "shape": [n, c, inh, inw]},
        "input1": {"dtype": "float32", "shape": [c, kh, kw]}
    }
    return ir, input_dict

def average_pooling(b, h, w, kh, kw, s, p):
    inh = (h - 1) * s + kh - 2 * p
    inw = (w - 1) * s + kw - 2 * p
    padh = inh + 2 * p
    padw = inw + 2 * p
    ir = f"pad[B, H0, W0] = input0[B, H0-{p}, W0-{p}].when([H0>={p}, H0<{inh+p}, W0>={p}, W0<{inw+p}], const(0.0).cast(input0[B, H0-{p}, W0-{p}].dtype())) where H0 in {padh}, W0 in {padw}; \
           output0[B, HO, WO] +=! pad[B, KH + HO*{s}, KW + WO*{s}] where HO in {h}, WO in {w}, KH in {kh}, KW in {kw};"
    input_dict = {
        "input0": {"dtype": "float32", "shape": [b, inh, inw]},
    }
    return ir, input_dict

def transpose(n, m):
    ir = "output0[N, M] = input0[M, N]"
    input_dict = {
        "input0": {"dtype": "float32", "shape": [n, m]},
    }
    return ir, input_dict

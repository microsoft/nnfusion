import copy
from functools import reduce
import operator
import re
from . import conv_expr
# from roller.op import *

config_global = {
    "topk": 16,
    "num_threads": 8,
    "smem_tiling": True,
    "reg_tiling": True,
    "st_align": False,
    "schedule_fuse": False,
    "use_tc": False,
    "shrink_tiny": True,
    "data_type": "float32",
    "padding_threshold_cap": 1.0
}

# warning: codegenR is not imported

def assert_large_parallelism(shape):
    if isinstance(shape, str): shape = [int(x) for x in shape.split(",")]
    tensor_size = reduce(operator.mul, shape, 1)
    assert(tensor_size >= 512) # roller is not efficient enough for small kernels

def conv_add_relu_parser(identifier):
    pattern = "Matched_Pattern\(Convolution-Add-Relu\)\[Convolution\[(\d+,\d+,\d+,\d+);(\d+,\d+,\d+,\d+);(\d+,\d+,\d+,\d+)floatfloatfloatLayout\{(\w+)\}Strides\{(\d+, \d+)\}Strides\{(\d+, \d+)\}CoordinateDiff\{(\d+, \d+)\}\]"
    inp, weight, out, layout, stride, dilation, padding = re.search(pattern, identifier).groups()
    assert_large_parallelism(out)
    F, N, HO, WO = [int(x) for x in out.split(",")]
    _, C, KH, KW = [int(x) for x in weight.split(",")]
    SH, SW = [int(x) for x in stride.split(",")]
    DH, DW = [int(x) for x in dilation.split(",")]
    PH, PW = [int(x) for x in padding.split(",")]
    assert(SH == SW)
    assert(DH == DW)
    assert(PH == PW)
    config = copy.deepcopy(config_global)
    config["schedule_fuse"] = True
    # if SH != 1 or DH !=1 or PH != 1: raise NotImplementedError

    func_name = f"fused_conv_bias_relu_expr_S{SH}D{DH}P{PH}"
    shape = (N, F, HO, WO, C, KH, KW)
    if layout == "CNHW":
        conv_expr.CNHW = True
    elif layout == "NCHW2CNHW":
        conv_expr.CNHW = False
    else:
        raise NotImplementedError(f"conv layout: {layout}")
    print("func_name", func_name)
    return func_name, shape, config

def conv_parser(identifier):
    pattern = "Convolution\[(\d+,\d+,\d+,\d+);(\d+,\d+,\d+,\d+);(\d+,\d+,\d+,\d+)floatfloatfloatLayout\{(\w+)\}Strides\{(\d+, \d+)\}Strides\{(\d+, \d+)\}CoordinateDiff\{(\d+, \d+)\}\]"
    inp, weight, out, layout, stride, dilation, padding = re.search(pattern, identifier).groups()
    assert_large_parallelism(out)
    F, N, HO, WO = [int(x) for x in out.split(",")]
    _, C, KH, KW = [int(x) for x in weight.split(",")]
    SH, SW = [int(x) for x in stride.split(",")]
    DH, DW = [int(x) for x in dilation.split(",")]
    PH, PW = [int(x) for x in padding.split(",")]
    assert(SH == SW)
    assert(DH == DW)
    assert(PH == PW)
    config = copy.deepcopy(config_global)
    config["schedule_fuse"] = True
    # if SH != 1 or DH !=1 or PH != 1: raise NotImplementedError

    func_name = f"fused_conv_expr_S{SH}D{DH}P{PH}"
    shape = (N, F, HO, WO, C, KH, KW)
    if layout == "CNHW":
        conv_expr.CNHW = True
    elif layout == "NCHW2CNHW":
        conv_expr.CNHW = False
    else:
        raise NotImplementedError(f"conv layout: {layout}")
    return func_name, shape, config

def avgpool_parser(identifier):
    pattern = "AvgPool\[(\d+,\d+,\d+,\d+);(\d+,\d+,\d+,\d+)floatfloatShape\{(\d+, \d+)\}Strides\{(\d+, \d+)\}Shape\{(\d+, \d+)\}Shape\{(\d+, \d+)\}]"
    inp, out, window_shape, window_stride, padding_below, padding_above = re.search(pattern, identifier).groups()
    assert_large_parallelism(out)
    N, F, HO, WO = [int(x) for x in out.split(",")]
    B = N * F
    _, _, HI, WI = [int(x) for x in inp.split(",")]

    KH, KW = [int(x) for x in window_shape.split(",")]
    SH, SW = [int(x) for x in window_stride.split(",")]
    assert(SH == SW)
    S = SH
    
    assert(padding_below == padding_above)
    PH, PW = [int(x) for x in padding_below.split(",")]
    assert(PH == PW)
    P = PH

    D = 1

    HI_comp = (HO - 1) * S + (KH - 1) * D + 1 - 2 * P
    WI_comp = (WO - 1) * S + (KW - 1) * D + 1 - 2 * P
    if HI_comp != HI or WI_comp != WI: raise NotImplementedError

    func_name = f"avgpool2d_expr_S{S}P{P}"
    shape = (B, HO, WO, KH, KW)
    return func_name, shape, copy.deepcopy(config_global)

def maxpool_parser(identifier):
    pattern = "MaxPool\[(\d+,\d+,\d+,\d+);(\d+,\d+,\d+,\d+)floatfloatShape\{(\d+, \d+)\}Strides\{(\d+, \d+)\}Shape\{(\d+, \d+)\}Shape\{(\d+, \d+)\}]"
    inp, out, window_shape, window_stride, padding_below, padding_above = re.search(pattern, identifier).groups()
    assert_large_parallelism(out)
    N, F, HO, WO = [int(x) for x in out.split(",")]
    B = N * F
    _, _, HI, WI = [int(x) for x in inp.split(",")]

    KH, KW = [int(x) for x in window_shape.split(",")]
    SH, SW = [int(x) for x in window_stride.split(",")]
    assert(SH == SW)
    S = SH
    
    assert(padding_below == padding_above)
    PH, PW = [int(x) for x in padding_below.split(",")]
    assert(PH == PW)
    P = PH

    D = 1

    HI_comp = (HO - 1) * S + (KH - 1) * D + 1 - 2 * P + 1
    WI_comp = (WO - 1) * S + (KW - 1) * D + 1 - 2 * P + 1
    if HI_comp != HI or WI_comp != WI: raise NotImplementedError

    func_name = f"maxpool2d_expr_S{S}P{P}"
    shape = (B, HO, WO, KH, KW)
    return func_name, shape, copy.deepcopy(config_global)

def dot_parser(identifier):
    pattern = "Dot\[(\d+,\d+);(\d+,\d+);(\d+,\d+)floatfloatfloat(\d)(\d)\]"
    A, B, out, transa, transb = re.search(pattern, identifier).groups()
    assert_large_parallelism(out)
    M, N = [int(x) for x in out.split(",")]
    transa = int(transa)
    transb = int(transb)
    A = [int(x) for x in A.split(",")]
    K = A[0] if transa else A[1]
    shape = (M, N, K)
    func_name = f"matmul_expr_{transa}{transb}"
    print(config_global)
    return func_name, shape, copy.deepcopy(config_global)

def conv_parser(identifier):
    pattern = "Convolution\[(\d+,\d+,\d+,\d+);(\d+,\d+,\d+,\d+);(\d+,\d+,\d+,\d+)floatfloatfloatLayout\{(\w+)\}Strides\{(\d+, \d+)\}Strides\{(\d+, \d+)\}CoordinateDiff\{(\d+, \d+)\}\]"
    inp, weight, out, layout, stride, dilation, padding = re.search(pattern, identifier).groups()
    assert_large_parallelism(out)
    F, N, HO, WO = [int(x) for x in out.split(",")]
    _, C, KH, KW = [int(x) for x in weight.split(",")]
    SH, SW = [int(x) for x in stride.split(",")]
    DH, DW = [int(x) for x in dilation.split(",")]
    PH, PW = [int(x) for x in padding.split(",")]
    assert(SH == SW)
    assert(DH == DW)
    assert(PH == PW)
    config = copy.deepcopy(config_global)
    config["schedule_fuse"] = True

    func_name = f"fused_conv_expr_S{SH}D{DH}P{PH}"
    shape = (N, F, HO, WO, C, KH, KW)
    if layout == "CNHW":
        conv_expr.CNHW = True
    elif layout == "NCHW2CNHW":
        conv_expr.CNHW = False
    else:
        raise NotImplementedError(f"conv layout: {layout}")
    print("func_name", func_name)
    return func_name, shape, config


def bmm_parser(identifier):
    pattern = "BatchMatMul\[(\d+,\d+);(\d+,\d+,\d+);(\d+,\d+,\d+)floatfloatfloat\]"
    if re.search(pattern, identifier) is not None:
        A, B, C = re.search(pattern, identifier).groups()
        assert_large_parallelism(C)
        m_a, k_a = [int(x) for x in A.split(",")]
        bs_b, k_b, n_b = [int(x) for x in B.split(",")]
        bs_c, m_c, n_c = [int(x) for x in C.split(",")]
        assert m_a == m_c
        assert k_a == k_b
        assert n_b == n_c
        assert bs_b == bs_c
        func_name = f"batch_matmul_bcast_a_expr"
        shape = (bs_b, m_a, n_b, k_a)
        config = copy.deepcopy(config_global)
        return func_name, shape, config
    # convert to pattern BatchMatMul[1,12,1,64;12,64,64;1,12,1,64floatfloatfloat]
    pattern = "BatchMatMul\[(\d+,\d+,\d+,\d+);(\d+,\d+,\d+);(\d+,\d+,\d+,\d+)floatfloatfloat\]"
    if re.search(pattern, identifier) is not None:
        A, B, C = re.search(pattern, identifier).groups()
        assert_large_parallelism(C)
        bs_a, m_a, k_a, n_a = [int(x) for x in A.split(",")]
        m_b, k_b, n_b = [int(x) for x in B.split(",")]
        bs_c, m_c, k_c, n_c = [int(x) for x in C.split(",")]
        assert bs_a == bs_c
        assert m_a == m_b
        assert m_b == m_c
        assert k_a == k_c
        assert n_a == k_b
        assert n_b == n_c
        func_name = f"batch_matmul_4d_3d_expr"
        shape = (bs_a, m_a, k_c, n_c, n_a)
        config = copy.deepcopy(config_global)
        # config["shrink_tiny"] = False
        return func_name, shape, config
    # convert to pattern BatchMatMul[1,12,64,64;1,12,64,1;1,12,64,1floatfloatfloat]
    pattern = "BatchMatMul\[(\d+,\d+,\d+,\d+);(\d+,\d+,\d+,\d+);(\d+,\d+,\d+,\d+)floatfloatfloat\]"
    if re.search(pattern, identifier) is not None:
        A, B, C = re.search(pattern, identifier).groups()
        assert_large_parallelism(C)
        bs_a, m_a, k_a, n_a = [int(x) for x in A.split(",")]
        bs_b, m_b, k_b, n_b = [int(x) for x in B.split(",")]
        bs_c, m_c, k_c, n_c = [int(x) for x in C.split(",")]
        assert bs_a == bs_b
        assert bs_b == bs_c
        assert m_a == m_b
        assert m_b == m_c
        assert k_a == k_c
        assert n_a == k_b
        assert n_b == n_c
        if identifier in ("BatchMatMul[1,12,64,64;1,12,64,1;1,12,64,1floatfloatfloat]", "BatchMatMul[64,12,64,64;64,12,64,1;64,12,64,1floatfloatfloat]"):
            # for reproducing the codegen with existing ansor tuning log
            func_name = f"batch_matmul_4d_4d_expr"
            shape = (bs_a, m_a, k_c, n_c, n_a)
        else:
            func_name = f"batch_matmul_expr"
            shape = (bs_a * m_a, k_c, n_c, n_a)
        config = copy.deepcopy(config_global)
        # config["shrink_tiny"] = False
        return func_name, shape, config
    raise NotImplementedError

id2func = {
    "Matched_Pattern(Convolution-Add-Relu)": conv_add_relu_parser,
    "Convolution": conv_parser,
    "AvgPool": avgpool_parser,
    "MaxPool": maxpool_parser,
    "Dot": dot_parser,
    "BatchMatMul": bmm_parser,
}

def get_func(identifier):
    kernel_name = identifier.split("[")[0]
    if kernel_name not in id2func:
        raise NotImplementedError(f"{kernel_name} with {identifier} is not yet supported")
    func_name, shape, config = id2func[kernel_name](identifier)
    return func_name, shape, config
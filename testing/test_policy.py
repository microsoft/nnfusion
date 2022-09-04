from ops import *
from memopt.graph import IRNode, OutputNode
import memopt
from memopt.fusion import DefaultPolicy
from arch import *
import numpy as np
from memopt.reference import get_subgraph_reference_outputs

def test_policy(ir, input_dict, name="test", check=False):
    expr = "- einstein_v2('{}', {})".format(ir, str(input_dict))
    A = IRNode([None for _ in input_dict], expr)
    output_nodes = [OutputNode(A)]
    policy = DefaultPolicy(output_nodes, V100())
    configs = policy.emit_config(10)

    compile_results = []
    cgen = memopt.CodeGenerator()
    for config in configs:
        cpresult = cgen.compile(output_nodes, config, "cuda", name="Fused")
        cpresult.append_host_call()
        compile_results.append(cpresult)
    memopt.utils.compile_and_load_parallel(compile_results)
    best_latency = 10000
    best = None
    values = []
    for cpresult in compile_results:
        print(cpresult.config)
        if cpresult.lib is None:
            latency = 10000
        else:
            latency = cpresult.profile()
        values.append(latency)
        if latency < best_latency:
            best_latency = latency
            best = cpresult
        print(latency)
    print(best.config)
    print(name ,"top1: {} \ttop10: {}".format(values[0], min(values)))
    print("-" * 80, flush=True)
    if check==True:
        out = best.get_example_outputs()
        ref_out = get_subgraph_reference_outputs(output_nodes)
        for a, b in zip(out, ref_out):
            diff = np.max(np.abs(a-b))
            print("value diff:", diff)
    return best, best_latency

c_lists = [
    ('C0', conv_nchw, [128, 128, 28, 28, 128, 3, 3, 1, 1, 1]),
    ('C1', conv_nchw, [128, 128, 28, 28, 128, 3, 3, 2, 1, 0]),
    ('C2', conv_nchw, [128, 256, 14, 14, 256, 3, 3, 2, 1, 0]),
    ('C3', conv_nchw, [128, 168, 42, 42, 168, 1, 1, 1, 1, 0]),
    ('C4', conv_nchw, [128, 512, 7, 7, 512, 3, 3, 1, 1, 1]),
    ('C5', conv_nchw, [128, 1024, 14, 14, 256, 1, 1, 1, 1, 0]),
    ('C6', conv_nchw, [128, 256, 14, 14, 1024, 1, 1, 1, 1, 0]),
    ('C7', conv_nchw, [128, 512, 14, 14, 1024, 1, 1, 1, 1, 0]),
    ('C8', conv_nchw, [128, 168, 21, 21, 1008, 1, 1, 1, 1, 0]),
    ('C9', conv_nchw, [128, 42, 83, 83, 42, 1, 1, 1, 1, 0]),
    ('C10', conv_nchw, [128, 672, 11, 11, 4032, 1, 1, 1, 1, 0]),
    ('C11', conv_nchw, [128, 512, 7, 7, 512, 3, 3, 2, 1, 0]),
    ('C12', conv_nchw, [128, 42, 83, 83, 96, 1, 1, 1, 1, 0]),
    ('C13', conv_nchw, [128, 42, 165, 165, 96, 1, 1, 1, 1, 0]),
    ('C14', conv_nchw, [128, 84, 83, 83, 168, 1, 1, 1, 1, 0]),
    ('C15', conv_nchw, [128, 336, 21, 21, 336, 1, 1, 1, 1, 0]),
    ('C16', conv_nchw, [128, 1024, 14, 14, 512, 1, 1, 2, 1, 0]),
    ('C17', conv_nchw, [128, 256, 56, 56, 64, 1, 1, 1, 1, 0]),
    ('C18', conv_nchw, [128, 64, 56, 56, 256, 1, 1, 1, 1, 0]),
    ('C19', conv_nchw, [128, 512, 28, 28, 128, 1, 1, 1, 1, 0]),
    ('C20', conv_nchw, [128, 128, 28, 28, 512, 1, 1, 1, 1, 0]),
    ('C21', conv_nchw, [128, 84, 42, 42, 168, 1, 1, 1, 1, 0]),
    ('C22', conv_nchw, [128, 256, 28, 28, 512, 1, 1, 1, 1, 0]),
    ('C23', conv_nchw, [128, 64, 56, 56, 64, 3, 3, 1, 1, 1]),
    ('C24', conv_nchw, [128, 672, 21, 21, 2016, 1, 1, 1, 1, 0]),
    ('C25', conv_nchw, [128, 2048, 7, 7, 512, 1, 1, 1, 1, 0]),
    ('C26', conv_nchw, [128, 512, 7, 7, 2048, 1, 1, 1, 1, 0]),
    ('C27', conv_nchw, [128, 84, 42, 42, 84, 1, 1, 1, 1, 0]),
    ('C28', conv_nchw, [128, 168, 42, 42, 336, 1, 1, 1, 1, 0]),
    ('C29', conv_nchw, [128, 672, 11, 11, 672, 1, 1, 1, 1, 0]),
    ('C30', conv_nchw, [128, 2048, 7, 7, 1024, 1, 1, 2, 1, 0]),
    ('C31', conv_nchw, [128, 336, 11, 11, 2016, 1, 1, 1, 1, 0]),
    ('C32', conv_nchw, [128, 336, 21, 21, 2016, 1, 1, 1, 1, 0]),
    ('C33', conv_nchw, [128, 336, 42, 42, 1008, 1, 1, 1, 1, 0]),
    ('C34', conv_nchw, [128, 64, 56, 56, 64, 1, 1, 1, 1, 0]),
    ('C35', conv_nchw, [128, 64, 112, 112, 3, 7, 7, 2, 1, 0]),
    ('C36', conv_nchw, [128, 96, 165, 165, 3, 3, 3, 2, 1, 0]),
    ('C37', conv_nchw, [128, 128, 56, 56, 256, 1, 1, 1, 1, 0]),
    ('C38', conv_nchw, [128, 256, 14, 14, 256, 3, 3, 1, 1, 0]),
    ('C39', conv_nchw, [128, 672, 11, 11, 2688, 1, 1, 1, 1, 0]),
    ('C40', conv_nchw, [128, 168, 42, 42, 1008, 1, 1, 1, 1, 0]),
    ('C41', conv_nchw, [128, 42, 83, 83, 96, 1, 1, 1, 1, 0]),
    ('C42', conv_nchw, [128, 512, 28, 28, 256, 1, 1, 2, 1, 0]),
    ('C43', conv_nchw, [128, 336, 21, 21, 1344, 1, 1, 1, 1, 0]),
]

m_lists = [
    ('M0', matmul_nn, [65536, 2, 1024]),
    ('M1', matmul_nn, [128, 4032, 1000]),
    ('M2', matmul_nn, [128, 2048, 1000]),
    ('M3', matmul_nn, [65536, 1024, 4096]),
    ('M4', matmul_nn, [65536, 1024, 1024]),
    ('M5', matmul_nn, [65536, 4096, 1024]),
    ('M6', matmul_nn, [65536, 30522, 1024]),
]

e_lists = [
    ('E0', relu, [227598336]),
    ('E1', relu, [6422528]),
    ('E2', relu, [25690112]),
    ('E3', relu, [12845056]),
    ('E4', relu, [334540800]),
    ('E5', relu, [75866112]),
    ('E6', relu, [41631744]),
    ('E7', relu, [102760448]),
    ('E8', relu, [102760448]),
    ('E9', relu, [12845056]),
    ('E10', relu, [51380224]),
    ('E11', relu, [25690112]),
    ('E12', relu, [113799168]),
    ('E13', relu, [10407936]),
    ('E14', relu, [148141056]),
    ('E15', relu, [25690112]),
    ('E16', relu, [37933056]),
    ('E17', relu, [18966528]),
    ('E18', relu, [62447616]),
    ('E19', relu, [3211264]),
    ('E20', relu, [12845056]),
    ('E21', relu, [74070528]),
    ('E22', relu, [75866112]),
    ('E23', relu, [146361600]),
    ('E24', relu, [37933056]),
    ('E25', relu, [18966528]),
    ('E26', relu, [37035264]),
    ('E27', relu, [51380224]),
]

r_lists = [
    ('R0', row_reduce, [65536, 1024]),
    ('R1', row_reduce, [65536, 1024]),
    ('R2', row_reduce, [516096, 121]),
    ('R3', row_reduce, [262144, 49]),
]

d_lists = [
    ('D0', dwconv_nchw, [128, 84, 42, 42, 5, 5, 2, 1, 2]),
    ('D1', dwconv_nchw, [128, 42, 83, 83, 5, 5, 1, 1, 2]),
    ('D2', dwconv_nchw, [128, 336, 21, 21, 5, 5, 1, 1, 2]),
    ('D3', dwconv_nchw, [128, 42, 83, 83, 5, 5, 2, 1, 2]),
    ('D4', dwconv_nchw, [128, 84, 42, 42, 7, 7, 2, 1, 3]),
    ('D5', dwconv_nchw, [128, 672, 11, 11, 3, 3, 1, 1, 1]),
    ('D6', dwconv_nchw, [128, 168, 42, 42, 5, 5, 1, 1, 2]),
    ('D7', dwconv_nchw, [128, 672, 11, 11, 5, 5, 2, 1, 2]),
    ('D8', dwconv_nchw, [128, 336, 21, 21, 3, 3, 1, 1, 1]),
    ('D9', dwconv_nchw, [128, 672, 11, 11, 7, 7, 2, 1, 3]),
    ('D10', dwconv_nchw, [128, 42, 83, 83, 7, 7, 1, 1, 3]),
    ('D11', dwconv_nchw, [128, 84, 42, 42, 7, 7, 1, 1, 3]),
    ('D12', dwconv_nchw, [128, 84, 42, 42, 5, 5, 1, 1, 2]),
    ('D13', dwconv_nchw, [128, 168, 42, 42, 3, 3, 1, 1, 1]),
    ('D14', dwconv_nchw, [128, 672, 11, 11, 7, 7, 1, 1, 3]),
    ('D15', dwconv_nchw, [128, 336, 21, 21, 5, 5, 2, 1, 2]),
    ('D16', dwconv_nchw, [128, 96, 83, 83, 5, 5, 2, 1, 2]),
    ('D17', dwconv_nchw, [128, 336, 21, 21, 7, 7, 1, 1, 3]),
    ('D18', dwconv_nchw, [128, 336, 21, 21, 7, 7, 2, 1, 3]),
    ('D19', dwconv_nchw, [128, 42, 83, 83, 3, 3, 1, 1, 1]),
    ('D20', dwconv_nchw, [128, 96, 83, 83, 7, 7, 2, 1, 3]),
    ('D21', dwconv_nchw, [128, 84, 42, 42, 3, 3, 1, 1, 1]),
    ('D22', dwconv_nchw, [128, 672, 11, 11, 5, 5, 1, 1, 2]),
]

p_lists = [
    ('P0', average_pooling, [21504, 42, 42, 1, 1, 2, 0]),
    ('P1', average_pooling, [86016, 11, 11, 3, 3, 2, 1]),
    ('P2', average_pooling, [5376, 83, 83, 3, 3, 1, 1]),
    ('P3', average_pooling, [129024, 21, 21, 1, 1, 2, 0]),
    ('P4', average_pooling, [43008, 21, 21, 3, 3, 2, 1]),
    ('P5', average_pooling, [10752, 42, 42, 3, 3, 2, 1]),
    ('P6', average_pooling, [86016, 11, 11, 3, 3, 1, 1]),
    ('P7', average_pooling, [12288, 83, 83, 1, 1, 2, 0]),
    ('P8', average_pooling, [258048, 11, 11, 1, 1, 2, 0]),
    ('P9', average_pooling, [5376, 83, 83, 3, 3, 2, 1]),
    ('P10', average_pooling, [10752, 42, 42, 3, 3, 1, 1]),
    ('P11', average_pooling, [43008, 21, 21, 3, 3, 1, 1]),
    ('P12', average_pooling, [21504, 42, 42, 3, 3, 1, 1]),
]

test_list = c_lists
for name, func, args in test_list:
    test_policy(*func(*args), name, False)

import memopt
import numpy as np
from memopt.arch import *
from memopt.config import Config, Stride
from memopt.graph import IRNode, OutputNode
from memopt.policy import *
from memopt.reference import get_subgraph_reference_outputs

from ops import *

arch = g3090()
def test_policy(ir, input_dict, name="test", check=True):
    expr = "- einstein_v2('{}', {})".format(ir, str(input_dict))
    A = IRNode([None for _ in input_dict], expr)
    output_nodes = [OutputNode(A)]
    policy = DefaultPolicy(output_nodes, arch)
    m, n = A.get_shape()
    if m % 8 == 0 and n % 8 == 0 and (n * m) % 256 == 0 and list(A.raxis.values())[0] % 16 == 0:
        A.add_tag("tensorCoreConfig", (0, 1))
        policy = TCPolicy(output_nodes, arch)
    configs = policy.emit_config(20)

    compile_results = []
    cgen = memopt.CodeGenerator()
    for config in configs:
        cpresult = cgen.compile(output_nodes, config, "cuda", kernel_name="Fused")
        compile_results.append(cpresult)
    memopt.utils.compile_and_load_parallel(compile_results, arch)
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
    print(best.code)
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
    ('C0', conv_nhwc_implicit_gemm, [128, 128, 28, 28, 128, 3, 3, 1, 1, 1]),
    ('C1', conv_nhwc_implicit_gemm, [128, 128, 28, 28, 128, 3, 3, 2, 1, 0]),
    ('C2', conv_nhwc_implicit_gemm, [128, 256, 14, 14, 256, 3, 3, 2, 1, 0]),
    ('C3', conv_nhwc_implicit_gemm, [128, 168, 42, 42, 168, 1, 1, 1, 1, 0]),
    ('C4', conv_nhwc_implicit_gemm, [128, 512, 7, 7, 512, 3, 3, 1, 1, 1]),
    ('C5', conv_nhwc_implicit_gemm, [128, 1024, 14, 14, 256, 1, 1, 1, 1, 0]),
    ('C6', conv_nhwc_implicit_gemm, [128, 256, 14, 14, 1024, 1, 1, 1, 1, 0]),
    ('C7', conv_nhwc_implicit_gemm, [128, 512, 14, 14, 1024, 1, 1, 1, 1, 0]),
    ('C8', conv_nhwc_implicit_gemm, [128, 168, 21, 21, 1008, 1, 1, 1, 1, 0]),
    ('C9', conv_nhwc_implicit_gemm, [128, 42, 83, 83, 42, 1, 1, 1, 1, 0]),
    ('C10', conv_nhwc_implicit_gemm, [128, 672, 11, 11, 4032, 1, 1, 1, 1, 0]),
    ('C11', conv_nhwc_implicit_gemm, [128, 512, 7, 7, 512, 3, 3, 2, 1, 0]),
    ('C12', conv_nhwc_implicit_gemm, [128, 42, 83, 83, 96, 1, 1, 1, 1, 0]),
    ('C13', conv_nhwc_implicit_gemm, [128, 42, 165, 165, 96, 1, 1, 1, 1, 0]),
    ('C14', conv_nhwc_implicit_gemm, [128, 84, 83, 83, 168, 1, 1, 1, 1, 0]),
    ('C15', conv_nhwc_implicit_gemm, [128, 336, 21, 21, 336, 1, 1, 1, 1, 0]),
    ('C16', conv_nhwc_implicit_gemm, [128, 1024, 14, 14, 512, 1, 1, 2, 1, 0]),
    ('C17', conv_nhwc_implicit_gemm, [128, 256, 56, 56, 64, 1, 1, 1, 1, 0]),
    ('C18', conv_nhwc_implicit_gemm, [128, 64, 56, 56, 256, 1, 1, 1, 1, 0]),
    ('C19', conv_nhwc_implicit_gemm, [128, 512, 28, 28, 128, 1, 1, 1, 1, 0]),
    ('C20', conv_nhwc_implicit_gemm, [128, 128, 28, 28, 512, 1, 1, 1, 1, 0]),
    ('C21', conv_nhwc_implicit_gemm, [128, 84, 42, 42, 168, 1, 1, 1, 1, 0]),
    ('C22', conv_nhwc_implicit_gemm, [128, 256, 28, 28, 512, 1, 1, 1, 1, 0]),
    ('C23', conv_nhwc_implicit_gemm, [128, 64, 56, 56, 64, 3, 3, 1, 1, 1]),
    ('C24', conv_nhwc_implicit_gemm, [128, 672, 21, 21, 2016, 1, 1, 1, 1, 0]),
    ('C25', conv_nhwc_implicit_gemm, [128, 2048, 7, 7, 512, 1, 1, 1, 1, 0]),
    ('C26', conv_nhwc_implicit_gemm, [128, 512, 7, 7, 2048, 1, 1, 1, 1, 0]),
    ('C27', conv_nhwc_implicit_gemm, [128, 84, 42, 42, 84, 1, 1, 1, 1, 0]),
    ('C28', conv_nhwc_implicit_gemm, [128, 168, 42, 42, 336, 1, 1, 1, 1, 0]),
    ('C29', conv_nhwc_implicit_gemm, [128, 672, 11, 11, 672, 1, 1, 1, 1, 0]),
    ('C30', conv_nhwc_implicit_gemm, [128, 2048, 7, 7, 1024, 1, 1, 2, 1, 0]),
    ('C31', conv_nhwc_implicit_gemm, [128, 336, 11, 11, 2016, 1, 1, 1, 1, 0]),
    ('C32', conv_nhwc_implicit_gemm, [128, 336, 21, 21, 2016, 1, 1, 1, 1, 0]),
    ('C33', conv_nhwc_implicit_gemm, [128, 336, 42, 42, 1008, 1, 1, 1, 1, 0]),
    ('C34', conv_nhwc_implicit_gemm, [128, 64, 56, 56, 64, 1, 1, 1, 1, 0]),
    ('C35', conv_nhwc_implicit_gemm, [128, 64, 112, 112, 3, 7, 7, 2, 1, 0]),
    ('C36', conv_nhwc_implicit_gemm, [128, 96, 165, 165, 3, 3, 3, 2, 1, 0]),
    ('C37', conv_nhwc_implicit_gemm, [128, 128, 56, 56, 256, 1, 1, 1, 1, 0]),
    ('C38', conv_nhwc_implicit_gemm, [128, 256, 14, 14, 256, 3, 3, 1, 1, 0]),
    ('C39', conv_nhwc_implicit_gemm, [128, 672, 11, 11, 2688, 1, 1, 1, 1, 0]),
    ('C40', conv_nhwc_implicit_gemm, [128, 168, 42, 42, 1008, 1, 1, 1, 1, 0]),
    ('C41', conv_nhwc_implicit_gemm, [128, 42, 83, 83, 96, 1, 1, 1, 1, 0]),
    ('C42', conv_nhwc_implicit_gemm, [128, 512, 28, 28, 256, 1, 1, 2, 1, 0]),
    ('C43', conv_nhwc_implicit_gemm, [128, 336, 21, 21, 1344, 1, 1, 1, 1, 0]),
]

test_list = c_lists
for name, func, args in test_list:
    test_policy(*func(*args), name, False)

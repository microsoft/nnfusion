import memopt
import numpy as np
import tvm
from memopt.arch import *
from memopt.config import Config, Stride
from memopt.debug import debug
from memopt.graph import IRNode, OutputNode
from memopt.policy import TCPolicy
from memopt.reference import get_subgraph_reference_outputs


def get_matmul_expr(n, k, m):
    ir = "output0[N, M] +=! input0[N, K] * input1[K, M]"
    input_dict = {
        "input0": {"dtype": "float16", "shape": [n, k]},
        "input1": {"dtype": "float16", "shape": [k, m]}
    }
    return ir, input_dict

def get_matmul_relu_expr(n, k, m):
    ir = "mediate[N, M] +=! input0[N, K] * input1[K, M]; output0[N, M] = mediate[N, M].call(`max`, [const(0).cast(mediate[N, M].dtype())]);"
    input_dict = {
        "input0": {"dtype": "float16", "shape": [n, k]},
        "input1": {"dtype": "float16", "shape": [k, m]}
    }
    return ir, input_dict

def run_all():
    target = tvm.target.cuda(arch="sm_70")
    n, m, k = 1920 * 1080, 64, 64
    ir, input_dict = get_matmul_relu_expr(n, k, m)
    expr = "- einstein_v2('{}', {})".format(ir, str(input_dict))
    L0 = IRNode([None, None], expr)
    L1 = IRNode([L0, None], expr)
    L2 = IRNode([L1, None], expr)
    L3 = IRNode([L2, None], expr)
    L4 = IRNode([L3, None], expr)
    L5 = IRNode([L4, None], expr)
    ir, input_dict = get_matmul_expr(n, k, 3)
    expr = "- einstein_v2('{}', {})".format(ir, str(input_dict))
    L6 = IRNode([L5, None], expr)
    output_nodes = [OutputNode(L6)]
    configs = {
        L0 : {"use_tc" : True, "strides" : {2 : Stride(72, 0)}, "block" : [128, 64], "warp": [64, 32], "wmma": [16, 16, 16], "rstep" : [64]},
        L1 : {"use_tc" : True, "strides" : {2 : Stride(72, 0)}, "block" : [128, 64], "warp": [64, 32], "wmma": [16, 16, 16], "rstep" : [64]},
        L2 : {"use_tc" : True, "strides" : {2 : Stride(72, 0)}, "block" : [128, 64], "warp": [64, 32], "wmma": [16, 16, 16], "rstep" : [64]},
        L3 : {"use_tc" : True, "strides" : {2 : Stride(72, 0)}, "block" : [128, 64], "warp": [64, 32], "wmma": [16, 16, 16], "rstep" : [64]},
        L4 : {"use_tc" : True, "strides" : {2 : Stride(72, 0)}, "block" : [128, 64], "warp": [64, 32], "wmma": [16, 16, 16], "rstep" : [64]},
        L5 : {"use_tc" : True, "strides" : {2 : Stride(72, 0)}, "block" : [128, 64], "warp": [64, 32], "wmma": [16, 16, 16], "rstep" : [64]},
        L6 : {"block": [128, 3], "thread": [128, 1], "rstep": [64]},
    }
    for node in configs:
        node.add_tag("tensorCoreConfig", (0, 1))
        configs[node] = Config().from_dict(configs[node]).complete_config(node)
    cpresult = memopt.CodeGenerator().compile(output_nodes, configs, target, kernel_name="Fused")
    cpresult.compile_and_load()
    print(cpresult.code)
    out = cpresult.get_example_outputs()
    print(cpresult.profile())
    print(out)
    # ref_out = get_subgraph_reference_outputs(output_nodes)
    # print(out, ref_out)
    # for a, b in zip(out, ref_out):
    #     diff = np.max(np.abs(a-b))
    #     print("value diff:", diff)

def run_search():
    target = tvm.target.cuda(arch="sm_70")
    target = tvm.target.cuda(arch="sm_70")
    n, m, k = 1920 * 1080, 64, 64
    ir, input_dict = get_matmul_relu_expr(n, k, m)
    expr = "- einstein_v2('{}', {})".format(ir, str(input_dict))
    L0 = IRNode([None, None], expr)
    L1 = IRNode([L0, None], expr)
    L2 = IRNode([L1, None], expr)
    L3 = IRNode([L2, None], expr)
    L4 = IRNode([L3, None], expr)
    L5 = IRNode([L4, None], expr)
    for node in (L0, L1, L2, L3, L4, L5):
        node.add_tag("tensorCoreConfig", (0, 1))
    ir, input_dict = get_matmul_expr(n, k, 3)
    expr = "- einstein_v2('{}', {})".format(ir, str(input_dict))
    L6 = IRNode([L5, None], expr)
    output_nodes = [OutputNode(L6)]
    policy = TCPolicy(output_nodes, V100())
    configs = policy.emit_config(10)
    compile_results = []
    cgen = memopt.CodeGenerator()
    for config in configs:
        cpresult = cgen.compile(output_nodes, config, target, kernel_name="Fused")
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
    print("top1: {} \ttop10: {}".format(values[0], min(values)))

if __name__ == "__main__":
    run_search()

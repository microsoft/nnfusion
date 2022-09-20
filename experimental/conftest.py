from memopt.fusion.tc import TCPolicy
from memopt.graph import IRNode, OutputNode
from memopt.debug import debug
from arch import *
from memopt.reference import get_subgraph_reference_outputs
import memopt
import tvm
import numpy as np

def nn(m, n, k):
    ir = "output0[M, N] +=! input0[M, K] * input1[K, N]"
    input_dict = {
        "input0": {"dtype": "float16", "shape": [m, k]},
        "input1": {"dtype": "float16", "shape": [k, n]}
    }
    return ir, input_dict

def nt(m, n, k):
    ir = "output0[M, N] +=! input0[M, K] * input1[N, K]"
    input_dict = {
        "input0": {"dtype": "float16", "shape": [m, k]},
        "input1": {"dtype": "float16", "shape": [n, k]}
    }
    return ir, input_dict


def bmm(b, m, n, k):
    ir = "output0[B, M, N] +=! input0[B, M, K] * input1[B, K, N]"
    input_dict = {
        "input0": {"dtype": "float16", "shape": [b, m, k]},
        "input1": {"dtype": "float16", "shape": [b, k, n]}
    }
    return ir, input_dict

def test(func, args, ax_m, ax_n):
    target = tvm.target.cuda(arch="sm_70")
    ir, input_dict = func(*args)
    expr = "- einstein_v2('{}', {})".format(ir, str(input_dict))
    A = IRNode([None for _ in input_dict], expr)
    A.add_tag("tensorCoreConfig", (ax_m, ax_n))
    output_nodes = [OutputNode(A)]
    policy = TCPolicy(output_nodes, V100())
    configs = policy.emit_config(10)
    compile_results = []
    cgen = memopt.CodeGenerator()
    for i, config in enumerate(configs):
        cpresult = cgen.compile(output_nodes, config, target, kernel_name="Fused"+str(i))
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
    compile_results = list(filter(lambda x:x.latency<10000, compile_results))
    compile_results = sorted(compile_results, key=lambda x:x.latency)
    ref_out = get_subgraph_reference_outputs(output_nodes)
    for best in compile_results:
        out = best.get_example_outputs()
        diff = 0
        for a, b in zip(out, ref_out):
            diff = max(diff, np.max(np.abs(a-b)))
        if diff > 1:
            print("Error: ", diff)
            continue
        break
    print(best.config)
    print(best.latency)

if __name__ == "__main__":
    # test(bmm, [64, 16, 512, 512], 1, 2)
    test(nn, [8192, 256, 256], 0, 1)

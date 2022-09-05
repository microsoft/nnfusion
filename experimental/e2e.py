from memopt.graph import IRNode, OutputNode
from memopt.debug import debug
from memopt.scheduler import Scheduler
from arch import *
from memopt.reference import get_subgraph_reference_outputs
import memopt
import tvm
import numpy as np

def get_matmul_expr(n, k, m):
    ir = "output0[N, M] +=! input0[N, K] * input1[K, M]"
    input_dict = {
        "input0": {"dtype": "float16", "shape": [n, k]},
        "input1": {"dtype": "float16", "shape": [k, m]}
    }
    return ir, input_dict

def get_matmul_suffix_expr(n, k, m):
    ir = "mediate[N, M] +=! input0[N, K] * input1[K, M]; output0[N, M] = mediate[N, M] * const(2.0).cast(mediate[N, M].dtype())"
    input_dict = {
        "input0": {"dtype": "float16", "shape": [n, k]},
        "input1": {"dtype": "float16", "shape": [k, m]}
    }
    return ir, input_dict

def run_single():
    target = tvm.target.cuda(arch="sm_70")
    n, m, k = 4096, 256, 256
    ir, input_dict = get_matmul_suffix_expr(n, m, k)
    tile = {"use_tc" : True, "N" : [64, 32, 16], "M" : [128, 64, 16], "K" : [64, 16]}
    expr = "- einstein_v2('{}', {})".format(ir, str(input_dict))
    A = IRNode([None for _ in input_dict], expr)
    sch, args = A.create_schedule(), A.args
    sch = Scheduler().rewrite_schedule(sch, tile, [])
    output_nodes = [OutputNode(A)]
    cpresult = memopt.CodeGenerator().compile(output_nodes, {A : tile}, "cuda", kernel_name="Fused")
    cpresult.append_host_call()
    cpresult.compile_and_load()
    # print(cpresult.profile())
    print(cpresult.code)
    out = cpresult.get_example_outputs()
    ref_out = get_subgraph_reference_outputs(output_nodes)
    for a, b in zip(out, ref_out):
        diff = np.max(np.abs(a-b))
        print("value diff:", diff)

def run_two():
    target = tvm.target.cuda(arch="sm_70")
    n, m, k = 1920 * 1080, 64, 64
    ir, input_dict = get_matmul_expr(n, m, k)
    tile = {"use_tc" : True, "N" : [128, 64, 16], "M" : [64, 32, 16], "K" : [64, 16]}
    expr = "- einstein_v2('{}', {})".format(ir, str(input_dict))
    A = IRNode([None for _ in input_dict], expr)
    B = IRNode([A, None], expr)
    output_nodes = [OutputNode(B)]
    cpresult = memopt.CodeGenerator().compile(output_nodes, {A : tile, B : tile}, "cuda", kernel_name="Fused")
    cpresult.append_host_call()
    cpresult.compile_and_load()
    print(cpresult.code)
    out = cpresult.get_example_outputs()
    print(cpresult.profile())
    ref_out = get_subgraph_reference_outputs(output_nodes)
    for a, b in zip(out, ref_out):
        diff = np.max(np.abs(a-b))
        print("value diff:", diff)

if __name__ == "__main__":
    run_two()

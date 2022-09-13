from memopt.fusion.config import TensorCoreExtraConfig
from memopt.graph import IRNode, OutputNode
from memopt.debug import debug
from memopt.scheduler import Scheduler
from arch import *
from memopt.reference import get_subgraph_reference_outputs
import memopt
import tvm
import numpy as np
from memopt.fusion import Config, Stride

def get_matmul_expr(m, n, k):
    ir = "output0[M, N] +=! input0[M, K] * input1[K, N]"
    input_dict = {
        "input0": {"dtype": "float16", "shape": [m, k]},
        "input1": {"dtype": "float16", "shape": [k, n]}
    }
    return ir, input_dict

def bmm(b, m, n, k):
    ir = "output0[B, M, N] +=! input0[B, M, K] * input1[B, K, N]"
    input_dict = {
        "input0": {"dtype": "float16", "shape": [b, m, k]},
        "input1": {"dtype": "float16", "shape": [b, k, n]}
    }
    return ir, input_dict


def run_single():
    target = tvm.target.cuda(arch="sm_70")
    m, n, k = 4096, 64, 64
    ir, input_dict = get_matmul_expr(m, n, k)
    tile = {
        "use_tc" : True, "strides" : {2 : Stride(72, 0)},
        "block" : [128, 64], "warp": [64, 32], "wmma": [16, 16, 16], "rstep" : [64]}
    tile = Config().from_dict(tile)
    expr = "- einstein_v2('{}', {})".format(ir, str(input_dict))
    A = IRNode([None for _ in input_dict], expr)
    tile.complete_config(A, 0, 1)
    sch = A.create_schedule()
    sch = Scheduler().rewrite_schedule(sch, tile, [])
    output_nodes = [OutputNode(A)]
    cpresult = memopt.CodeGenerator().compile(output_nodes, {A : tile}, target, kernel_name="Fused")
    # print(cpresult.code)
    cpresult.append_host_call()
    cpresult.compile_and_load()
    print(cpresult.profile(2))
    out = cpresult.get_example_outputs()
    ref_out = get_subgraph_reference_outputs(output_nodes)
    for a, b in zip(out, ref_out):
        diff = np.max(np.abs(a-b))
        print("value diff:", diff)

def run_bmm():
    target = tvm.target.cuda(arch="sm_70")
    b, m, n, k = 4, 4096, 64, 64
    ir, input_dict = bmm(b, m, n, k)
    tile = {
        "use_tc" : True, "strides" : {2 : Stride(72, 0)},
        "block" : [1, 128, 64], "warp": [1, 64, 32], "wmma": (16, 16, 16), "rstep" : [64]}
    tile = Config().from_dict(tile)
    expr = "- einstein_v2('{}', {})".format(ir, str(input_dict))
    A = IRNode([None for _ in input_dict], expr)
    tile.complete_config(A, 1, 2)
    sch = A.create_schedule()
    sch = Scheduler().rewrite_schedule(sch, tile, [])
    output_nodes = [OutputNode(A)]
    cpresult = memopt.CodeGenerator().compile(output_nodes, {A : tile}, target, kernel_name="Fused")
    print(cpresult.code)
    cpresult.append_host_call()
    cpresult.compile_and_load()
    print(cpresult.profile(2))
    out = cpresult.get_example_outputs()
    ref_out = get_subgraph_reference_outputs(output_nodes)
    for a, b in zip(out, ref_out):
        diff = np.max(np.abs(a-b))
        print("value diff:", diff)
    print(out, ref_out)

if __name__ == "__main__":
    run_bmm()

import memopt
import numpy as np
import tvm
from memopt.graph import IRNode
from memopt.layout import *
from memopt.schedule.cutlass_intrin import *
from memopt.te_utils import *
from memopt.utils import CompileResult
from tvm import te

tvm.register_func("tvm_callback_cuda_compile", override=True)(lambda x:"")

def gemm(n, m, k):
    """TVM expression for vector add"""
    A = te.placeholder((n, k), dtype="float16", name='input0')
    B = te.placeholder((k, m), dtype="float16", name='input1')
    K = te.reduce_axis((0, k))
    C = te.compute((n, m), lambda i, j: te.sum(A[i,K]*B[K,j], axis=[K]), name='output0')
    return A, B, C

def add_bias(n, m):
    C = te.placeholder((n, m), dtype="float16", name="input0")
    bias = te.placeholder((m, ), dtype="float16", name="input1")
    act = te.compute((m, n), lambda i, j: C[i,j] + bias[j], name='output0')
    return C, bias, act

n, m, k = 8192, 3072, 768

arg1 = gemm(n, m, k)
arg2 = add_bias(n, m)

args = connect_tensor_graph(arg1, arg2, {arg2[0]:arg1[2]})

node = IRNode([None for _ in range(3)], args)
print(node.infer_reverse([64, 128]))
# from memopt.config import Config
# from memopt.schedule.te_reduce import TEReduceScheduler as Scheduler

# config = Config().from_dict({'block': [128, 64], 'thread': [8, 16], 'rstep': [32]})

# s = Scheduler(args, config)
# s.schedule()
# src = s.build("cuda")
# print(src)
import numpy as np
import torch
import ctypes
import tvm
from tvm import te
from memopt import CodeGenerator

import memopt

# Defined in file: ./chapter_common_operators/matmul.md
def matmul(n, m, l):
    """Return the computing expression of matrix multiplication
    A : n x l matrix
    B : l x m matrix
    C : n x m matrix with C = A B
    """
    k = te.reduce_axis((0, l), name='k')
    A = te.placeholder((n, l), name='A')
    B = te.placeholder((l, m), name='B')
    C = te.compute((n, m),
                    lambda x, y: te.sum(A[x, k] * B[k, y], axis=k),
                    name='C')
    return A, B, C

target = tvm.target.cuda(arch="sm_61")
n, m, k = 4096, 128, 128
X, K, Y = matmul(n, m, k)
sch = te.create_schedule(Y.op)
cgen = CodeGenerator()
codegen_dict = {'k': [32, 1], 'x': [16, 4], 'y': [16, 4]}
sch = cgen.rewrite_schedule(sch, codegen_dict, True, True, target_stage="C")

with memopt.Scope(sch) as scope:
    kernel_code = memopt.build_op(sch, [X, K, Y], target, [], [], name="MyMatMul", global_kernel=True)
    # kernel_code = mod.imported_modules[0].get_source()
    # print(scope.block_size)
    code = memopt.utils.append_host_call(kernel_code, scope.block_size, scope.grid_size, 3, "MyMatMul", True)
    lib = memopt.utils.compile_and_load(code)
    lib.function.restype = ctypes.c_float
    print(code)
    a = torch.randn(n, k).cuda()
    b = torch.randn(k, m).cuda()
    c = torch.randn(n, m).cuda()
    ret = lib.function(
        ctypes.c_void_p(a.data_ptr()),
        ctypes.c_void_p(b.data_ptr()),
        ctypes.c_void_p(c.data_ptr())
    )
    print(ret)
    memopt.utils.ctypesCloseLibrary(lib)

import numpy as np
import torch
import ctypes
import tvm
from tvm import te
from memopt import CodeGenerator

import memopt
from memopt.tvm_ops import tvm_matmul

target = tvm.target.cuda(arch="sm_61")
n, m, k = 4096, 64, 64
X, K, Y = tvm_matmul(n, m, k)
sch = te.create_schedule(Y.op)
cgen = CodeGenerator()
codegen_dict = {'k': [16, 1], 'x': [4, 4, 2], 'y': [32, 2]}
sch = cgen.recursive_schedule_up(sch, codegen_dict, tile_blacklist=[])

with memopt.Scope(sch) as scope:
    kernel_code = memopt.build_op(sch, [X, K, Y], target, [], [], name="MyMatMul", global_kernel=True)
    code = memopt.utils.append_host_call(kernel_code, scope.block_size, scope.grid_size, 3, "MyMatMul", True)
    lib = memopt.utils.compile_and_load(code)
    lib.function.restype = ctypes.c_float
    print(code)
    a = torch.randn(n, k).cuda()
    b = torch.randn(k, m).cuda()
    c = torch.zeros(n, m).cuda()
    ret = lib.function(
        ctypes.c_void_p(a.data_ptr()),
        ctypes.c_void_p(b.data_ptr()),
        ctypes.c_void_p(c.data_ptr())
    )
    ref = torch.matmul(a, b)
    print(memopt.utils.profile(lib, [X, K, Y]))
    memopt.utils.ctypesCloseLibrary(lib)

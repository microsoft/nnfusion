import numpy as np
import torch
import ctypes
import tvm
from tvm import te
from memopt import Scheduler, Config

import memopt
from memopt.tvm_ops import tvm_matmul

target = tvm.target.cuda(arch="sm_61")
n, m, k = 4096, 64, 64
args = tvm_matmul(n, m, k)
sch = te.create_schedule(args[-1].op)
cgen = Scheduler()
codegen_dict = Config().from_dict({"block": [32, 64], "thread": [4, 32], "rstep" : [16], "step" : [2, 1]})
sch = cgen.rewrite_schedule(sch, codegen_dict, shared_inputs=[])

with memopt.Scope(sch) as scope:
    kernel_code = memopt.tvm_build.tvm_build(sch, args, target, [], [], name="MatMul", global_kernel=True)
    cp = memopt.utils.CompileResult(None, kernel_code, scope.block_size, scope.grid_size, "MatMul", args)
    cp.append_host_call()
    print(kernel_code)
    lib = cp.compile_and_load()
    print(scope.exteral_shared_memroy_size)
    a = torch.randn(n, k).cuda()
    b = torch.randn(k, m).cuda()
    c = torch.zeros(n, m).cuda()
    lib.call(
        ctypes.c_void_p(a.data_ptr()),
        ctypes.c_void_p(b.data_ptr()),
        ctypes.c_void_p(c.data_ptr())
    )
    torch.cuda.synchronize()
    ref = torch.matmul(a, b)
    print(torch.max(torch.abs(c - ref)))
    print(cp.profile())

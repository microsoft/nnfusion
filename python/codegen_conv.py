import numpy as np
from memopt.tvm_ops import tvm_conv
import tvm
from tvm import te
from memopt import Scheduler
import memopt

target = tvm.target.cuda(arch="sm_61")
oc, ic, n, k, s = 64, 64, 64, 3, 1

codegen_dict = {'k': [9, 1], 'nn': [1, 1], 'ff': [16, 2], 'yy': [4, 2], 'xx': [4, 1]}
args = tvm_conv(1, ic, n, n, oc, k, s, 1)
sch = te.create_schedule(args[-1].op)
sch = Scheduler().recursive_schedule_up(sch, codegen_dict, shared_inputs=[])

with memopt.Scope(sch) as scope:
    kernel_code = memopt.build_op(sch, args, target, [-1], [], name="MyPointWiseConv", global_kernel=True)
    cp = memopt.utils.CompileResult(None, kernel_code, scope.block_size, scope.grid_size, "MyPointWiseConv", args)
    cp.append_host_call()
    print(kernel_code)
    lib = cp.compile_and_load()
    print(scope.exteral_shared_memroy_size)

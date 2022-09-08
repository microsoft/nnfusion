from memopt.tvm_ops import tvm_conv
import tvm
from tvm import te
from memopt import Scheduler, Config
from memopt.tvm_build import tvm_build
import memopt

target = tvm.target.cuda(arch="sm_61")
oc, ic, n, k, s = 64, 64, 64, 3, 1

codegen_dict = Config().from_dict({"block" : [1, 16, 8, 4], "thread": [1, 16, 4, 4], "rstep": [9]})

args = tvm_conv(1, ic, n, n, oc, k, s, 1)
sch = te.create_schedule(args[-1].op)
sch = Scheduler().rewrite_schedule(sch, codegen_dict, shared_inputs=[])

with memopt.Scope(sch) as scope:
    kernel_code = tvm_build(sch, args, target, [], [], name="MyPointWiseConv", global_kernel=True)
    cp = memopt.utils.CompileResult(None, kernel_code, scope.block_size, scope.grid_size, "MyPointWiseConv", args)
    cp.append_host_call()
    print(kernel_code)
    lib = cp.compile_and_load()
    print(scope.exteral_shared_memroy_size)

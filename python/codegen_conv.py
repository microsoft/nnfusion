import numpy as np
from memopt.tvm_ops import tvm_conv
import tvm
from tvm import te
from memopt import CodeGenerator
import memopt
from arch import V100
from op import ConvOp
from policy import ConstructionPolicyV2

target = tvm.target.cuda(arch="sm_61")
oc, ic, n, k, s = 64, 64, 64, 3, 1
# arch = V100()
# op = ConvOp(1, ic, oc, k, s, n, n, 1, "SAME")
# Tiling_Policy = ConstructionPolicyV2(op, arch, saxis_names, raxis_names)
# configs = Tiling_Policy.emit_config_without_trails(10)[:10]
# for config in configs:
#     print(config.to_codegen_dict())
codegen_dict = {'k': [9, 1], 'nn': [1, 1], 'ff': [16, 2], 'yy': [4, 2], 'xx': [4, 1]}
X, K, Y = tvm_conv(1, ic, n, n, oc, k, s, 1)
sch = te.create_schedule(Y.op)
saxis_names = [axis.var.name for axis in sch[Y].op.axis]
raxis_names = [axis.var.name for axis in sch[Y].op.reduce_axis]
cgen = CodeGenerator()
sch = cgen.rewrite_schedule(sch, codegen_dict, True, True, tile_blacklist=[])

with memopt.Scope(sch) as scope:
    kernel_code = memopt.build_op(sch, [X, K, Y], target, [Y.name], [], name="MyPointWiseConv", global_kernel=True)
    code = memopt.utils.append_host_call(kernel_code, scope.block_size, scope.grid_size, 3, "MyPointWiseConv", True)
    print(code)
    lib = memopt.utils.compile_and_load(code)
    print(scope.exteral_shared_memroy_size)

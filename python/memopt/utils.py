import tvm
from .modify_input_pass import modify_input_pass
from .modify_output_pass import modify_output_pass
from .scope import Scope
import numpy as np

_tvm_default_name = "default_function_kernel0"

def build_op(sch, args, target, sm_outputs=[], sm_inputs=[], name=_tvm_default_name, global_kernel=True):
    passes = [
        (0, modify_output_pass),
        (0, modify_input_pass),
    ]
    assert(isinstance(sm_outputs, (tuple, list)))
    assert(isinstance(sm_inputs, (tuple, list)))
    with tvm.transform.PassContext(config={"tir.add_lower_pass": passes}), \
        Scope(sch) as scope:
        scope.shared_mem_outputs = sm_outputs
        scope.shared_mem_inputs = sm_inputs
        mod = tvm.build(sch, args, target=target)

        src = mod.imported_modules[0].get_source()
        index = src.index(_tvm_default_name)
        if global_kernel:
            prefix = "__global__ void __launch_bounds__(%d) " % np.prod(scope.block_size)
        else:
            prefix = "__device__ void "
        src = prefix + name + src[index+len(_tvm_default_name):]
    return src

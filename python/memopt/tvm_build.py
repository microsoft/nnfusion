from .modify_input_pass import modify_input_pass
from .modify_output_pass import modify_output_pass
from .debug_pass import debug_pass, get_kernel_info_pass
from .scope import get_scope

import regex as re
import tvm

_tvm_default_name = "default_function_kernel0"
_type_map = {"float32" : "float", "float16" : "half"}

def get_valid_name(var):
    if var.name.find(".") >= 0:
        name = var.name[:var.name.index(".")]
    else:
        name = var.name
    return name if var.value_index == 0 else name + str(var.value_index)

def tvm_build(sch, args, target, sm_outputs=[], sm_inputs=[], name=_tvm_default_name, global_kernel=True):
    scope = get_scope()
    passes = [
        (0, modify_output_pass),
        (0, modify_input_pass),
        (4, get_kernel_info_pass),
    ]
    disabled_pass = ["tir.StorageRewrite"] if sm_inputs else []
    assert(isinstance(sm_outputs, (tuple, list)))
    assert(isinstance(sm_inputs, (tuple, list)))
    func_args = ", ".join(["{}* __restrict__ {}".format(_type_map[var.dtype], get_valid_name(var)) for var in args])
    with tvm.transform.PassContext(
        config={"tir.add_lower_pass": passes}, disabled_pass=disabled_pass):
        scope.shared_mem_outputs = sm_outputs
        scope.shared_mem_inputs = sm_inputs

        old_entry = tvm.get_global_func("tvm_callback_cuda_compile")
        tvm.register_func("tvm_callback_cuda_compile", override=True)(lambda x:"")
        mod = tvm.build(sch, args, target=target)
        tvm.register_func("tvm_callback_cuda_compile", override=True)(old_entry)

        src = mod.imported_modules[0].get_source()
        index = src.rindex(_tvm_default_name)
        index = src.index("{", index)
        if global_kernel:
            prefix = "__global__ void __launch_bounds__(%d) " % np.prod(scope.block_size)
        else:
            prefix = "__device__ void "
            func_args += ", char* shared"
        src = prefix + name + "({}) ".format(func_args) + src[index:]
        # removing shared memory allocation
        for var in scope.shared_mem_inputs:
            s_var = var+"_shared"
            src = re.sub(r"__shared__ (\w+) {}\[\d+\];".format(s_var), r"\1* {} = {};".format(s_var, var), src, 1)
        if not global_kernel:
            for var, offset in scope.interal_shared_memory_offset.items():
                src = re.sub(r"__shared__ (\w+) {}\[\d+\];".format(var), r"\1* {} = (\1*)(shared+{});".format(var, offset), src, 1)
    return src

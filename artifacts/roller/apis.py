from roller.op import Op
from roller.policy import *
from roller.arch import dispatch_arch
from roller.codegen.op_impl.codegenR import CodeGeneratorR

def get_config_space(op, device_name):
    print('[debug] devname =', device_name)

    shape = [item.dom.extent.value for item in op.axis] + [item.dom.extent.value for item in op.reduce_axis]
    roller_op = Op(op, shape, op.output(0).dtype, False)
    roller_arch = dispatch_arch(device_name)

    # print(roller_op.IODependent())

    if roller_op.IODependent():
        policy = ConstructionPolicyRT(roller_op, roller_arch, True, True, False, 1.0, True)
    else:
        policy = ConstructionPolicyPlainRT(roller_op, roller_arch, True, True, True, 1.0)

    rprogs = policy.emit_config_without_trails(10)
    candidates = [(rprog, policy.get_align_info(rprog, roller_arch, True, True, target_stage=op.output(0).name, st_align=False), roller_arch) for rprog in rprogs]
    return candidates

def apply_config(op, sched, config):
    rprog, align_info, roller_arch = config
    print('[debug] config =', rprog.Dump())

    cgen = CodeGeneratorR()
    cgen.rewrite_schedule(sched, rprog, True, True, target_stage=op.output(0).name, align_info=align_info, bank_size=roller_arch.smem_bank_size)

    return sched

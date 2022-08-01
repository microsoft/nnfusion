from roller.op import Op
from roller.policy import *
from roller.arch import dispatch_arch
from roller.codegen.op_impl.codegenR import CodeGeneratorR

# matmul
smem_tiling = True
reg_tiling = True

# element wise
# smem_tiling = False
# reg_tiling = False

padding_threshold_cap = 1.0

fuse = False
schedule_fuse = False

st_align = False
keep_tiny = False
use_tc = False


def get_config_space(op, device_name):
    print('[debug] op info = ', op)
    print('[debug] devname =', device_name)

    shape = [item.dom.extent.value for item in op.axis
             ] + [item.dom.extent.value for item in op.reduce_axis]
    roller_op = Op(op, shape, op.output(0).dtype, use_tc)
    roller_arch = dispatch_arch(device_name)

    print('[debug] is IODependent: {}'.format(roller_op.IODependent()))

    if roller_op.IODependent():
        policy = ConstructionPolicyRT(roller_op, roller_arch, smem_tiling,
                                      reg_tiling, st_align, padding_threshold_cap, not keep_tiny)
    else:
        policy = ConstructionPolicyPlainRT(roller_op, roller_arch, smem_tiling,
                                           reg_tiling, st_align, padding_threshold_cap)

    rprogs = policy.emit_config_without_trails(10)
    candidates = []
    for rprog in rprogs:
        if fuse or schedule_fuse:
            # TODO: write_stage
            align_info = policy.get_align_info_fuse(
                rprog,
                roller_arch,
                smem_tiling,
                reg_tiling,
                target_stage=op.output(0).name,
                write_stage=op.output(0).name,
                st_align=st_align)
        else:
            align_info = policy.get_align_info(rprog,
                                               roller_arch,
                                               smem_tiling,
                                               reg_tiling,
                                               target_stage=op.output(0).name,
                                               st_align=st_align)
        candidates.append((rprog, align_info, roller_arch))
    return candidates


def apply_config(op, sched, config):
    rprog, align_info, roller_arch = config
    print('[debug] config =', rprog.Dump())

    cgen = CodeGeneratorR()
    if fuse or schedule_fuse:
        cgen.rewrite_schedule_fuse(sched,
                                   rprog,
                                   smem_tiling,
                                   reg_tiling, [], [op.output(0)],
                                   op.output(0),
                                   target_stage=op.output(0).name,
                                   write_stage=op.output(0).name,
                                   align_info=align_info,
                                   bank_size=roller_arch.smem_bank_size)
    else:
        cgen.rewrite_schedule(sched,
                              rprog,
                              smem_tiling,
                              reg_tiling,
                              target_stage=op.output(0).name,
                              align_info=align_info,
                              bank_size=roller_arch.smem_bank_size)

    return sched

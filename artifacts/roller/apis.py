from roller.op import Op
from roller.policy import *
from roller.arch import dispatch_arch
from roller.codegen.op_impl.codegenR import CodeGeneratorR
from roller.utils import schedule_tensorcore
import tvm

smem_tiling = True
reg_tiling = True

padding_threshold_cap = 1.0

fuse = False
schedule_fuse = False

st_align = False
keep_tiny = False
use_tc = False

def extract_shape_info(op):
    out_tensor = op.output(0)

    def extractor(op):
        print(op)
        return [item.dom.extent.value for item in op.axis
             ] + [item.dom.extent.value for item in op.reduce_axis]

    if '_unpad' in out_tensor.name:
        assert(len(op.input_tensors) == 1)
        return extractor(op.input_tensors[0].op)
    else:
        return extractor(op)


def get_config_space(op, device_name):
    assert(op.num_outputs == 1)

    print('[debug] devname =', device_name)
    print('[debug] op info = ', op)

    # todo
    shape = extract_shape_info(op)
    roller_op = Op(op, shape, op.output(0).dtype, use_tc)
    roller_arch = dispatch_arch(device_name)

    is_IODependent = roller_op.IODependent()
    if not is_IODependent:
        global smem_tiling, reg_tiling
        smem_tiling, reg_tiling = False, False

    print('[debug] is IODependent: {}'.format(is_IODependent))


    if is_IODependent:
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

            assert(len(roller_op.output_tensors) == 2)
            if '_unpad' in roller_op.output_tensors[0].name:
                out_tensor, write_tensor = roller_op.output_tensors[1], roller_op.output_tensors[0]
            else:
                out_tensor, write_tensor = roller_op.output_tensors[0], roller_op.output_tensors[1]

            align_info = policy.get_align_info_fuse(
                rprog,
                roller_arch,
                smem_tiling,
                reg_tiling,
                target_stage=out_tensor.name,
                write_stage=write_tensor.name,
                st_align=st_align)
        else:
            align_info = policy.get_align_info(rprog,
                                               roller_arch,
                                               smem_tiling,
                                               reg_tiling,
                                               target_stage=op.output(0).name,
                                               st_align=st_align)
        candidates.append((rprog, align_info, roller_arch, roller_op))
    return candidates


def apply_config(op, sched, config):
    rprog, align_info, roller_arch, roller_op = config
    print('[debug] config =', rprog.Dump())

    if use_tc:
        assert(op.num_outputs == 1)
        return schedule_tensorcore(sched, rprog, op.output(0))

    cgen = CodeGeneratorR()
    if fuse or schedule_fuse:
        ori_in, pad_in = [], []

        for tensor in roller_op.input_tensors:
            if '_pad' in tensor.name:
                pad_in.append(tensor)
            else:
                ori_in.append(tensor)

        if '_unpad' in roller_op.output_tensors[0].name:
            out_tensor, write_tensor = roller_op.output_tensors[1], roller_op.output_tensors[0]
        else:
            out_tensor, write_tensor = roller_op.output_tensors[0], roller_op.output_tensors[1]


        cgen.rewrite_schedule_fuse(sched,
                                   rprog,
                                   smem_tiling,
                                   reg_tiling, pad_in, [out_tensor],
                                   write_tensor,
                                   target_stage=out_tensor.name,
                                   write_stage=write_tensor.name,
                                   align_info=align_info,
                                   bank_size=roller_arch.smem_bank_size,
                                   ori_in=ori_in)

    else:
        cgen.rewrite_schedule(sched,
                              rprog,
                              smem_tiling,
                              reg_tiling,
                              target_stage=op.output(0).name,
                              align_info=align_info,
                              bank_size=roller_arch.smem_bank_size)

    return sched

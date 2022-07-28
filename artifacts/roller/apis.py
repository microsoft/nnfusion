from roller.op import Op, MatmulOp
from roller.policy import *
# from roller.test_config import matmul_expr
from roller.arch import V100
from roller.codegen.op_impl.codegenR import CodeGeneratorR

def get_config_space(op, device_name):
    print('[debug] devname =', device_name)

    roller_op = Op(op, (4096, 4096, 4096), 'float32', False)
    # TODO
    roller_arch = V100()

    # print(roller_op.IODependent())

    if roller_op.IODependent():
        policy = ConstructionPolicyRT(roller_op, roller_arch, True, True, False, 1.0, True)
    else:
        policy = ConstructionPolicyPlainRT(roller_op, roller_arch, True, True, True, 1.0)

    # TODO: Replace following with Roller's config list produced by rProg
    # candidates = [{"N": 1, "M": 2}, {"N": 2, "M": 1}]
    # return candidates

    rprogs = policy.emit_config_without_trails(10)
    return rprogs

def apply_config(op, sched, config):
    # import tvm
    print('[debug] config =', config.Dump())
    # print(tvm.lower(sched, op.input_tensors, simple_mode=True))
    # TODO: Replace following with Roller's rewrite_schedule()
    # sched[op].bind(op.axis[0], tvm.te.thread_axis('blockIdx.x'))

    roller_op = Op(op, (4096, 4096, 4096), 'float32', False)
    roller_arch = V100()

    if roller_op.IODependent():
        policy = ConstructionPolicyRT(roller_op, roller_arch, True, True, False, 1.0, True)
    else:
        policy = ConstructionPolicyPlainRT(roller_op, roller_arch, True, True, True, 1.0)

    align_info = policy.get_align_info(config, roller_arch, True, True, target_stage=op.output(0).name, st_align=False)

    # print(align_info)
    # print(op.output(0).name)
    cgen = CodeGeneratorR()
    # s = tvm.te.create_schedule(op)
    # print(tvm.lower(sched, op.input_tensors, simple_mode=True))
    cgen.rewrite_schedule(sched, config, True, True, target_stage=op.output(0).name, align_info=align_info, bank_size=roller_arch.smem_bank_size)
    # print(tvm.lower(sched, op.input_tensors, simple_mode=True))

    # in_tensors = op.input_tensors
    # out_tensor = op.output(0)
    # print([in_tensors[0], in_tensors[1]] + [out_tensor])
    # func = tvm.build(sched, [in_tensors[0], in_tensors[1]] + [out_tensor], 'cuda', name='template_op')
    # print(func.imported_modules[0].get_source())

    return sched

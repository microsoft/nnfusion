from roller.op import Op
from roller.policy import *
from roller.arch import dispatch_arch
from roller.codegen.op_impl.codegenR import CodeGeneratorR
from roller.utils import schedule_tensorcore, extract_shape_info
import tvm


def get_config_space(op, device_name):
  assert op.num_outputs == 1, 'ERROR: input op\'s output num is {}, expected 1'.format(
      op.num_outputs)

  print('[debug] devname =', device_name)
  print('[debug] op info = ', op)

  setting = {
      'smem_tiling': True,
      'reg_tiling': True,
      'padding_threshold_cap': 4.0,
      'fuse': False,
      'schedule_fuse': False,
      'st_align': False,
      'keep_tiny': False,
      'use_tc': False,
  }

  # TODO: align shape
  shape = extract_shape_info(op)
  roller_op = Op(op, shape, op.output(0).dtype, setting['use_tc'])
  roller_arch = dispatch_arch(device_name)

  if len(roller_op.output_tensors) > 1:
    setting['schedule_fuse'] = True

  is_IODependent = roller_op.IODependent()

  if len(roller_op.RAxis()) == 0:
    global smem_tiling, reg_tiling
    setting['smem_tiling'], setting['reg_tiling'] = False, False

  print('[debug] is IODependent: {}'.format(is_IODependent))

  if is_IODependent and len(roller_op.RAxis()) > 0:
    policy = ConstructionPolicyRT(roller_op, roller_arch,
                                  setting['smem_tiling'], setting['reg_tiling'],
                                  setting['st_align'],
                                  setting['padding_threshold_cap'],
                                  not setting['keep_tiny'])
  else:
    policy = ConstructionPolicyPlainRT(roller_op, roller_arch,
                                       setting['smem_tiling'],
                                       setting['reg_tiling'],
                                       setting['st_align'],
                                       setting['padding_threshold_cap'])

  rprogs = policy.emit_config_without_trails(10)
  candidates = []
  for rprog in rprogs:
    if setting['fuse'] or setting['schedule_fuse']:
      assert len(roller_op.output_tensors) == 2

      if '_unpad' in roller_op.output_tensors[0].name:
        out_tensor, write_tensor = roller_op.output_tensors[
            1], roller_op.output_tensors[0]
      else:
        out_tensor, write_tensor = roller_op.output_tensors[
            0], roller_op.output_tensors[1]

      align_info = policy.get_align_info_fuse(
          rprog,
          roller_arch,
          setting['smem_tiling'],
          setting['reg_tiling'],
          target_stage=out_tensor.name,
          write_stage=write_tensor.name,
          st_align=setting['st_align'])
    else:
      align_info = policy.get_align_info(
          rprog,
          roller_arch,
          setting['smem_tiling'],
          setting['reg_tiling'],
          target_stage=op.output(0).name,
          st_align=setting['st_align'])
    candidates.append(
        (rprog, align_info, roller_arch, roller_op, setting.copy()))
  return candidates


def apply_config(op, sched, config):
  rprog, align_info, roller_arch, roller_op, setting = config
  print('[debug] config =', rprog.Dump())

  if setting['use_tc']:
    assert op.num_outputs == 1, 'use_tc = True expected 1 output, while get {}'.format(
        op.num_outputs)
    return schedule_tensorcore(sched, rprog, op.output(0))

  cgen = CodeGeneratorR()
  if setting['fuse'] or setting['schedule_fuse']:
    ori_in, pad_in = [], []

    for tensor in roller_op.input_tensors:
      if '_pad' in tensor.name:
        pad_in.append(tensor)
      else:
        ori_in.append(tensor)

    if '_unpad' in roller_op.output_tensors[0].name:
      out_tensor, write_tensor = roller_op.output_tensors[
          1], roller_op.output_tensors[0]
    else:
      out_tensor, write_tensor = roller_op.output_tensors[
          0], roller_op.output_tensors[1]

    cgen.rewrite_schedule_fuse(
        sched,
        rprog,
        setting['smem_tiling'],
        setting['reg_tiling'],
        pad_in, [out_tensor],
        write_tensor,
        target_stage=out_tensor.name,
        write_stage=write_tensor.name,
        align_info=align_info,
        bank_size=roller_arch.smem_bank_size,
        ori_in=ori_in)

  else:
    cgen.rewrite_schedule(
        sched,
        rprog,
        setting['smem_tiling'],
        setting['reg_tiling'],
        target_stage=op.output(0).name,
        align_info=align_info,
        bank_size=roller_arch.smem_bank_size)

  return sched

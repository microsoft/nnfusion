from roller import Op, CodeGeneratorR
from roller.policy import *
from roller.arch import dispatch_arch
from roller.utils import schedule_tensorcore, extract_shape_info
from roller.config import rTile, rProg
import tvm
import json

__all__ = [
    'get_config_space',
    'apply_config',
]


def get_config_space(op, device_name, use_tc):
  assert op.num_outputs == 1, 'ERROR: input op\'s output num is {}, expected 1'.format(
      op.num_outputs)

  print('[debug] devname =', device_name)
  print('[debug] op info = ', op)

  setting = {
      'smem_tiling': True,
      'reg_tiling': True,
      'padding_threshold_cap': 1,
      'fuse': False,
      'schedule_fuse': False,
      'st_align': False,
      'keep_tiny': False,
      'use_tc': use_tc,
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

  # if is_IODependent and len(roller_op.RAxis()) > 0:
  #   policy = PolicyRTV2(roller_op, roller_arch, setting['smem_tiling'],
  #                 setting['reg_tiling'], setting['st_align'],
  #                 setting['padding_threshold_cap'],
  #                 not setting['keep_tiny'])
  # else:
  #   policy = PolicyPlainRTV2(roller_op, roller_arch, setting['smem_tiling'],
  #                          setting['reg_tiling'], setting['st_align'],
  #                          setting['padding_threshold_cap'])
  # rprogs = policy.emit_config_without_trails(20)

  if len(roller_op.RAxis()) > 0:
    rprogs1 = PolicyRTV2(roller_op, roller_arch, setting['smem_tiling'],
                  setting['reg_tiling'], setting['st_align'],
                  setting['padding_threshold_cap'],
                  not setting['keep_tiny']).emit_config_without_trails(10)
    rprogs2 = PolicyPlainRTV2(roller_op, roller_arch, setting['smem_tiling'],
                           setting['reg_tiling'], setting['st_align'],
                           setting['padding_threshold_cap']).emit_config_without_trails(10)
    rprogs = rprogs1 + rprogs2
  else:
    rprogs = PolicyPlainRTV2(roller_op, roller_arch, setting['smem_tiling'],
                           setting['reg_tiling'], setting['st_align'],
                           setting['padding_threshold_cap']).emit_config_without_trails(10)


  candidates = []
  for rprog in rprogs:
    candidates.append((rprog.Dump(), json.dumps(setting), is_IODependent))
  return candidates


def build_rprog4str(rprog_str, roller_op, roller_arch):
  rprog = rProg(roller_arch.num_level, roller_op)

  json_dict = json.loads(rprog_str)
  for item in json_dict.items():
    mem_level, rtile_json = item
    rtile_dict = json.loads(rtile_json)
    shape = rtile_dict['tile'] + rtile_dict['step']
    cur_rtile = rTile(roller_op.tvm_op, shape, roller_op.SAxis(),
                      roller_op.RAxis(), roller_op.GetTvmOutTensor())
    rprog.AddTile(int(mem_level), cur_rtile)

  return rprog


def apply_config(op, sched, config, device_name):
  rprog_str, setting_str, is_IODependent_str = config
  is_IODependent = bool(is_IODependent_str)
  setting = json.loads(setting_str)

  shape = extract_shape_info(op)
  roller_op = Op(op, shape, op.output(0).dtype, setting['use_tc'])
  roller_arch = dispatch_arch(device_name)

  rprog = build_rprog4str(rprog_str, roller_op, roller_arch)
  if is_IODependent and len(roller_op.RAxis()) > 0:
    policy = PolicyRTV2(roller_op, roller_arch, setting['smem_tiling'],
                      setting['reg_tiling'], setting['st_align'],
                      setting['padding_threshold_cap'],
                      not setting['keep_tiny'])
  else:
    policy = PolicyPlainRT(roller_op, roller_arch, setting['smem_tiling'],
                           setting['reg_tiling'], setting['st_align'],
                           setting['padding_threshold_cap'])

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

  print('[debug] config =', rprog.Dump())

  if setting['use_tc'] and op.output(0).dtype=='float16':
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

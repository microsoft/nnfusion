# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from tvm import te
import logging
import sys, time, subprocess
import json
import os
import re


def _schedule_single(attrs, output, op_name, tail_op):
  cfg, s = attrs.auto_config, attrs.scheduler

  def cache_local(output):
    if not tail_op:
      OL = s.cache_write(output, 'local')
    else:
      s[output].set_scope('local')
      OL, output = output, s.outputs[0].output(0)
    return output, OL

  s.cache_local = cache_local

  if len(output.op.reduce_axis) > 0:
    num_inputs = len(s[output].op.input_tensors)
    tile_algo = cfg.define_knob(f'M{op_name}T',
                                [True, False]) if num_inputs > 1 else False

    if tile_algo:
      from .algo_tiling import schedule_branch
      return schedule_branch(attrs, output, f'M{op_name}:')
    else:
      from .algo_reduce import schedule_branch
      return schedule_branch(attrs, output, f'M{op_name}:', tail_op)

  from .algo_format import schedule_branch
  return schedule_branch(attrs, output, f'F{op_name}:')


def schedule(attrs):
  config = os.environ.get('CONFIG', '').strip()
  step = int(os.environ.get('STEP', '0'))
  device_name = os.environ.get('DEV_NAME', 'V100')
  use_tc = os.environ.get('USETC', '0')
  attrs.advanced_sched = config or step > 0
  tail_op, explicit_ops = None, [x for x in attrs.explicit_ops]

  import traceback
  op = explicit_ops[-1]
  op_body = str(op.body)
  match_case = '\[reduce\(combiner=comm_reducer\(result=\[\(x \+ y\)\], lhs=\[x\], rhs=\[y\], identity_element=\[0h\]\), source=\[\([a-zA-Z0-9]+\[.+,.+\]\*[a-zA-Z0-9]+\[.+,.+\]\)\], init='
  x = re.search(match_case, op_body)
  use_tc = x != None
  
  try:
    from roller import apis
    from antares.common import AntaresGlobal
    if hasattr(AntaresGlobal,
               'auto_config') and AntaresGlobal.auto_config._candidate:
      config_case = json.loads(os.environ['MY_SPACE'])[
          AntaresGlobal.auto_config._candidate['RollerCase']]
    else:
      config_list = apis.get_config_space(
          op, device_name=device_name, use_tc=use_tc)
      config_case = attrs.auto_config.define_knob(
          f'RollerCase',
          list(config_list),
          init_vals=list(range(len(config_list))))
      os.environ['MY_SPACE'] = json.dumps(config_list)
    attrs.scheduler = apis.apply_config(
        op, sched=attrs.scheduler, config=config_case, device_name=device_name)
    return
  except:
    traceback.print_exc()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(1)

  if (len(explicit_ops) > 1 and not explicit_ops[-1].output(0).op.reduce_axis):
    fuse_tail = attrs.auto_config.define_knob(f'FU', [False, True])
    tail_op = explicit_ops[-1]
    if fuse_tail:
      tail_op.is_fused, explicit_ops = True, explicit_ops[:-1]
    else:
      tail_op.is_fused = False

  for rank, op in enumerate(reversed(explicit_ops)):
    _schedule_single(attrs, op.output(0), op.name,
                     tail_op if rank == 0 else None)
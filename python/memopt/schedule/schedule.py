from typing import Dict, List

from tvm import te

from ..config import Config, Stride
from .scheduler_base import get_compute_ops, seperate_reduce_ops
from .te_elementwise import *
from .te_reduce import *
from .te_reduce_interthread import *
from .te_wmma import *
from .tir_mma import TIRCutlassMMAScheduler
from .tir_simt import *


def schedule(args: List[te.Tensor], config: Config, shared_inputs: List[te.Tensor] = [],
        shared_inputs_strides: Dict[te.Tensor, Stride] = {}, shared_outputs = []):
    ops = get_compute_ops(args)
    reduces_ops, _ = seperate_reduce_ops(ops)
    if len(reduces_ops) == 0:
        template = TEElementWiseScheduler
    elif config.use_tc:
        A_ax_m, A_ax_k, B_ax_k, B_ax_n, C_ax_m, C_ax_n = config.tc_extra_conf.tc_axis
        if config.warp[C_ax_m]%32==0 and config.warp[C_ax_n]%32==0:
            template = TEWarpMMAScheduler
        else:
            template = TEWarpMMAScheduler
    elif any([t > 1 for t in config.reduce_thread]):
        template = TEReduceInterThreadScheduler
    else:
        template = TEReduceScheduler

    scheduler = template(args, config)

    scheduler.shared_inputs = shared_inputs
    scheduler.shared_outputs = shared_outputs
    scheduler.shared_inputs_strides = {tensor: Stride() for tensor in shared_inputs}
    scheduler.shared_inputs_strides.update(shared_inputs_strides)

    scheduler.make_passes()
    scheduler.schedule()

    return scheduler

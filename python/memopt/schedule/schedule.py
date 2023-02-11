from typing import Dict, List

from tvm import te

from ..config import Config, Stride
from .scheduler_base import get_compute_ops, seperate_reduce_ops
from .te_elementwise import *
from .te_reduce import *
from .te_reduce_interthread import *
from .te_wmma import *


def schedule(args: List[te.Tensor], config: Config, shared_inputs: List[te.Tensor] = [],
        shared_inputs_strides: Dict[te.Tensor, Stride] = {}, shared_outputs = []):
    ops = get_compute_ops(args)
    reduces_ops, _ = seperate_reduce_ops(ops)
    if len(reduces_ops) == 0:
        template = TEElementWiseScheduler
    elif config.use_tc:
        template = TEWarpMMAScheduler
    elif any([t > 1 for t in config.reduce_thread]):
        template = TEReduceInterThreadScheduler
    else:
        template = TEReduceScheduler

    scheduler = template(args)

    scheduler.shared_inputs = shared_inputs
    scheduler.shared_outputs = shared_outputs
    scheduler.shared_inputs_strides = {tensor: Stride() for tensor in shared_inputs}
    scheduler.shared_inputs_strides.update(shared_inputs_strides)

    return scheduler.schedule(config)

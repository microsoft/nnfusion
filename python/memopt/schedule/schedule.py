from typing import Dict, List

from tvm import te

from ..config import Config, Stride
from ..te_utils import get_compute_ops, seperate_reduce_ops
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
        if config.use_tc >= "80":
            arch_m, arch_n = 16, 8
        elif config.use_tc >= "70":
            arch_m, arch_n = 32, 32
        else:
            raise ValueError(config.use_tc)
        use_cutlass_warp_mma = True
        use_cutlass_warp_mma &= config.warp[C_ax_m]%arch_m==0 and config.warp[C_ax_n]%arch_n==0
        # cutlass_warp_mma currently don't support shared inputs as it uses pipeline approaches
        use_cutlass_warp_mma &= len(shared_inputs)==0
        # cutlass_warp_mma don't support batched mm inside a block
        for idx, value in enumerate(config.block):
            if idx not in [C_ax_m, C_ax_n]: use_cutlass_warp_mma &= value==1
        # use pipeline for large reduce ops
        reduce_size = np.prod([int(ax.dom.extent) for ax in reduces_ops[0].reduce_axis])
        use_cutlass_warp_mma &= reduce_size > 64
        template = TIRCutlassMMAScheduler if use_cutlass_warp_mma else TEWarpMMAScheduler
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

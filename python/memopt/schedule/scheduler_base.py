from typing import List, Tuple, Union

import tvm
from tvm import te, tir

from ..config import Config
from ..IRpass import *


def get_compute_ops(args: List[te.Tensor]) -> List[te.ComputeOp]:
    cur_stack = args.copy()
    ops = set()
    while len(cur_stack) > 0:
        tensor = cur_stack.pop(0)
        if not isinstance(tensor.op, te.ComputeOp): continue
        if tensor.op in ops: continue
        ops.add(tensor.op)
        for input_tensor in tensor.op.input_tensors:
            cur_stack.append(input_tensor)
    return list(ops)

def seperate_reduce_ops(ops: List[te.ComputeOp]) -> Tuple[List[te.ComputeOp], List[te.ComputeOp]]:
    # find out the op with reduce axis
    reduce_ops = []
    none_reduce_ops = []
    for op in ops:
        if len(op.reduce_axis) > 0:
            reduce_ops.append(op)
        else:
            none_reduce_ops.append(op)
    return reduce_ops, none_reduce_ops

class SchedulerBase:
    def __init__(self, args: List[te.Tensor], config: Config) -> None:
        self.args = args
        self.config = config
        self.ops = get_compute_ops(self.args)

        output_ops = set()
        for arg in self.args:
            if arg.op in self.ops:
                output_ops.add(arg.op)
        assert(len(output_ops) > 0)
        if len(output_ops) > 1:
            raise RuntimeError("Op with multiple output stages should create a proxy output.")
        self.output_op = output_ops.pop()

        reduce_ops, self.none_reduce_ops = seperate_reduce_ops(self.ops)
        if len(reduce_ops) == 0:
            self.reduce_op = None
        elif len(reduce_ops) == 1:
            self.reduce_op = reduce_ops[0]
        else:
            raise RuntimeError("Can't' schedule Op with multiple reduce stages.")

        self.block_size = [1, 1, 1] # blockDim.xyz
        self.compute_grid_size()
        self.sche = self.create_schedule()
        self.passes = []

        self.shared_inputs = []
        self.shared_outputs = []
        self.shared_inputs_strides = []

    def _is_from_shared(self, tensor):
        if tensor in self.shared_inputs:
            return True
        return any(map(self._is_from_shared, tensor.op.input_tensors))

    def compute_grid_size(self):
        tile_shape = self.config.block
        node_shape = self.output_op.output(0).shape
        size = 1
        for t, n in zip(tile_shape, node_shape):
            size *= (n + t - 1) // t
        self.grid_size = [int(size), 1, 1]

    def make_passes(self) -> None:
        return

    def create_schedule(self) -> Union[te.Schedule, tir.Schedule]:
        raise NotImplementedError()

    def build(self, target) -> str:
        raise NotImplementedError()

    def schedule(self) -> Union[te.Schedule, tir.Schedule]:
        raise NotImplementedError()

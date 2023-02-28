from typing import List, Tuple

from tvm import te, tir


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
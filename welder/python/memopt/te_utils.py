from typing import Callable, Dict, List, Tuple

from tvm import _ffi, te, tir


def create_proxy_output(tensors: List[te.Tensor]):
    layout = ", ".join([ax.var.name for ax in tensors[0].op.axis])
    sandbox = {"args" : tensors}
    exec("func=lambda {}: [op[{}] for op in args]".format(layout, layout), sandbox)
    proxy_output = te.compute(tensors[0].shape, sandbox["func"], name="output_proxy")
    return list(proxy_output)

def pre_order_traverse(tensors: List[te.Tensor], func: Callable):
    visited = set()
    def _traverse(tensor):
        if tensor in visited: return
        visited.add(tensor)
        for input_tensor in tensor.op.input_tensors:
            _traverse(input_tensor)
        func(tensor)

    for tensor in tensors:
        _traverse(tensor)

def get_compute_ops(tensors: List[te.Tensor]) -> List[te.ComputeOp]:
    ops = []
    def functor(tensor):
        if isinstance(tensor.op, te.ComputeOp):
            ops.append(tensor.op)
    pre_order_traverse(tensors, functor)
    return ops

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

def tensor_replace_input(tensor: te.Tensor, name: str=None, rmap: Dict[te.Tensor, te.Tensor]={}) -> te.Tensor:
    op = tensor.op
    if name:
        if isinstance(op, te.PlaceholderOp):
            op = _ffi.get_global_func("te.PlaceholderOp")(name, op.shape, op.dtype)
        elif isinstance(op, te.ComputeOp):
            op = _ffi.get_global_func("te.ComputeOp")(name, op.tag, op.attrs, op.axis, op.body)
        else:
            raise NotImplementedError()
    if len(rmap) > 0:
        op = _ffi.get_global_func("te.OpReplaceInputs")(op, rmap)
    tensor = _ffi.get_global_func("te.TensorReplaceOp")(tensor, op)
    return tensor


def connect_tensor_graph(a: List[te.Tensor], b: List[te.Tensor], connection):
    tensor_map = {}
    return_args = []
    input_idx, output_idx, mid_idx = 0, 0, 0

    def functor(tensor: te.Tensor):
        nonlocal input_idx, output_idx, mid_idx
        is_input = isinstance(tensor.op, te.PlaceholderOp) and tensor not in connection.keys()
        is_output = not isinstance(tensor.op, te.PlaceholderOp) and tensor in a + b and tensor not in connection.values()

        if is_input:
            name = "input" + str(input_idx)
            input_idx += 1
        elif is_output:
            name = "output" + str(output_idx)
            output_idx += 1
        else:
            name = "mediate" + str(mid_idx)
            mid_idx += 1

        rmap = {}
        for it in tensor.op.input_tensors:
            dst = connection[it] if it in connection else it
            rmap[it] = tensor_map[dst]

        replaced = tensor_replace_input(tensor, name, rmap)
        tensor_map[tensor] = replaced

        if is_input:
            if tensor not in connection.keys():
                return_args.append(replaced)
        elif is_output:
            return_args.append(replaced)

    pre_order_traverse(a + b, functor)

    return return_args

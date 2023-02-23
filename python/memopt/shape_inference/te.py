from collections import OrderedDict
from typing import List

from tvm import arith, te, tir

from .common import InputShapeInference, Statement


def get_compute_ops(args: List[te.Tensor]) -> List[te.ComputeOp]:
    cur_stack = args.copy()
    ops = []
    while len(cur_stack) > 0:
        tensor = cur_stack.pop(0)
        if not isinstance(tensor.op, te.ComputeOp): continue
        if tensor.op in ops: continue
        ops.append(tensor.op)
        for input_tensor in tensor.op.input_tensors:
            cur_stack.append(input_tensor)
    return ops

def walk_indice(expr):
    if isinstance(expr, tir.expr.BinaryOpExpr):
        a = walk_indice(expr.a)
        b = walk_indice(expr.b)
        if a is not None and b is not None:
            return expr
        else:
            return None
    elif isinstance(expr, tir.expr.ConstExpr):
        return expr
    elif isinstance(expr, tir.Var):
        return expr
    elif isinstance(expr, tir.ProducerLoad):
        return None
    elif isinstance(expr, (tir.Call, tir.Cast)):
        return None
    else:
        raise Exception('Unhandled node type in walk_indice(): %s' % expr)

def _get_analyzer_by_te(op: te.ComputeOp) -> Statement:
    output = str(op.name)
    var_map = OrderedDict()
    range_map = OrderedDict()
    for ax in op.axis:
        var_map[str(ax.var.name)] = ax.var
    for ax in op.reduce_axis:
        var_map[str(ax.var.name)] = ax.var
        range_map[ax.var] = arith.ConstIntBound(int(ax.dom.min), int(ax.dom.min + ax.dom.extent - 1))

    dependent_region = {}

    def fvisit(x):
        if not isinstance(x, tir.ProducerLoad): return

        tensor_name = str(x.producer.name)
        if tensor_name not in dependent_region:
            dependent_region[tensor_name] = []
        index = []
        for indice, shape_limit in zip(x.indices, x.producer.shape):
            expr = walk_indice(indice)
            if expr is None:
                expr = te.var("undefined") % shape_limit
            index.append(expr)
        dependent_region[tensor_name].append(index)

    for expr in op.body:
        tir.stmt_functor.post_order_visit(expr, fvisit=fvisit)

    return Statement(output, dependent_region, var_map, range_map)


def get_analyzer_by_te(args : List[te.Tensor]) -> InputShapeInference:
    deps = [_get_analyzer_by_te(op) for op in reversed(get_compute_ops(args))]

    return InputShapeInference(deps)

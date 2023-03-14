from typing import Dict, List

from tvm import arith, te, tir

from ..te_utils import get_compute_ops


class Statement():
    def __init__(self, op: te.ComputeOp, dependent_region: Dict[str, List[tir.PrimExpr]]):
        self.op = op
        self.dependent_region = dependent_region

        self.reverse_bound_inference = {}

    def make_reverse(self, input_name: str):
        if len(self.op.reduce_axis) > 0:
            return None
        if len(self.dependent_region[input_name]) != 1:
            return None
        indices = self.dependent_region[input_name][0]
        iter_map_range = {ax.var: ax.dom for ax in self.op.axis}
        iter_map_result = arith.detect_iter_map(indices, iter_map_range, check_level=arith.iter_affine_map.IterMapLevel.Surjective)
        if len(iter_map_result.errors) > 0:
            return None
        output_iter = [te.var("i"+str(i)) for i in range(len(indices))]
        results = arith.iter_affine_map.inverse_affine_iter_map(iter_map_result.indices, output_iter)
        output_indices = [results[ax.var] for ax in self.op.axis]
        return output_iter, output_indices

    def infer_reverse(self, input_name: str, shape: List[arith.ConstIntBound]):
        if input_name not in self.reverse_bound_inference:
            self.reverse_bound_inference[input_name] = self.make_reverse(input_name)
        if self.reverse_bound_inference[input_name] is None:
            raise RuntimeError(f"Cannot reverse infer bound from {input_name}")
        output_iter, output_indices = self.reverse_bound_inference[input_name]
        ana = arith.Analyzer()
        for var, bound in zip(output_iter, shape):
            ana.update(var, bound)
        bounds = [ana.const_int_bound(index) for index in output_indices]
        return bounds

def _merge_two_bounds(x: arith.ConstIntBound, y: arith.ConstIntBound):
    return arith.ConstIntBound(min(x.min_value, y.min_value), max(x.max_value, y.max_value))

class InputShapeInference():
    def __init__(self, deps: List[Statement]):
        self.deps = deps

    def infer(self, shape: Dict[str, List[arith.ConstIntBound]], rstep: Dict[str, int]={}):
        shape = shape.copy()
        ana = arith.Analyzer()
        for dep in reversed(self.deps):
            for ax, bound in zip(dep.op.axis, shape[dep.op.name]):
                ana.update(ax.var, bound)
            for ax in dep.op.reduce_axis:
                bound = arith.ConstIntBound(int(ax.dom.min), int(ax.dom.min + ax.dom.extent - 1))
                if ax.var.name in rstep:
                    bound = arith.ConstIntBound(int(ax.dom.min), int(ax.dom.min + min(ax.dom.extent, rstep[ax.var.name]) - 1))
                ana.update(ax.var, bound)
            for name, regions in dep.dependent_region.items():
                for region in regions:
                    bounds = [ana.const_int_bound(index) for index in region]
                    if name in shape: # simply merge two bounds
                        bounds = [_merge_two_bounds(x, y) for x, y in zip(shape[name], bounds)]
                    shape[name] = bounds

        for name, bounds in shape.items():
            shape[name] = [c.max_value - c.min_value + 1 for c in bounds]
        return shape

    def infer_reverse(self, input_name: str, shape):
        for dep in self.deps:
            if input_name in dep.dependent_region:
                shape = dep.infer_reverse(input_name, shape)
                input_name = dep.op.name
        return input_name, shape

    def get_input_exprs(self, output_exprs):
        result = output_exprs.copy()
        ana = arith.Analyzer()
        for dep in reversed(self.deps):
            for ax, expr in zip(dep.op.axis, result[dep.op.name]):
                ana.bind(ax.var, expr)
            for ax in dep.op.reduce_axis:
                ana.bind(ax.var, 0)
            for name, regions in dep.dependent_region.items():
                if name in result:
                    continue
                region = regions[0]
                input_expr = [ana.simplify(index) for index in region]
                result[name] = input_expr
        return result

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
    return Statement(op, dependent_region)


def get_analyzer_by_te(args : List[te.Tensor]) -> InputShapeInference:
    deps = [_get_analyzer_by_te(op) for op in get_compute_ops(args)]

    return InputShapeInference(deps)

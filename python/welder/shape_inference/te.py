from typing import Dict, List, Tuple

from tvm import arith, te, tir

from ..te_utils import get_compute_ops


class Statement():
    def __init__(self, op: te.ComputeOp):
        self.op = op
        self.dependent_region = _extract_dependent_region(op)

        self.reverse_bound_inference = {}

    def make_reverse(self, input_name: str, input_iter: List[tir.PrimExpr]):
        if len(self.op.reduce_axis) > 0:
            return None
        if len(self.dependent_region[input_name]) != 1:
            return None
        indices = self.dependent_region[input_name][0]
        iter_map_range = {ax.var: ax.dom for ax in self.op.axis}
        iter_map_result = arith.detect_iter_map(indices,
                                                iter_map_range,
                                                check_level=arith.iter_affine_map.IterMapLevel.Surjective,
                                                simplify_trivial_iterators=False
                                                )
        if len(iter_map_result.errors) > 0:
            return None
        results = arith.iter_affine_map.inverse_affine_iter_map(iter_map_result.indices, input_iter)
        output_indices = []
        for ax in self.op.axis:
            if ax.var in results:
                output_indices.append(results[ax.var])
            else:
                # not Bijective mapping case
                output_indices.append(te.var("undefined") % ax.dom.extent)
        return output_indices

def _merge_two_bounds(x: arith.ConstIntBound, y: arith.ConstIntBound):
    return arith.ConstIntBound(min(x.min_value, y.min_value), max(x.max_value, y.max_value))

class InputShapeInference():
    def __init__(self, deps: List[Statement]):
        self.deps = deps
        self.target_mapping = {}
        self.reduce_axes = []
        for dep in self.deps:
            for ax in dep.op.reduce_axis:
                self.reduce_axes.append(ax)

    def construct_dependency_target(self, targets: Tuple[str]):
        if targets in self.target_mapping:
            return self.target_mapping[targets]
        name2dep = {dep.op.name: dep for dep in self.deps}
        mapping = {}
        input_vars = []
        for target in targets:
            vars = [te.var(f"i{i}") for i in range(len(name2dep[target].op.axis))]
            input_vars.append(vars)
            mapping[target] = [vars]

        ana = arith.Analyzer()

        for dep in self.deps:
            for name in dep.dependent_region:
                if name not in mapping: continue
                indices = mapping[name][0]
                output_indices = dep.make_reverse(name, indices)
                if dep.op.name not in mapping:
                    mapping[dep.op.name] = [output_indices]
                elif not region_exist_in_list(output_indices, mapping[dep.op.name]):
                    raise Exception(f"Cannot perform shape inference from {targets}")

        for dep in reversed(self.deps):
            indices_list = mapping[dep.op.name]
            ax_vars = [ax.var for ax in dep.op.axis]
            for input_name, regions in dep.dependent_region.items():
                if input_name not in mapping:
                    mapping[input_name] = []
                for indices in indices_list:
                    for region in regions:
                        vmap = {k: v for k, v in zip(ax_vars, indices)}
                        region = [ana.simplify(tir.stmt_functor.substitute(ax, vmap)) for ax in region]
                        if not region_exist_in_list(region, mapping[input_name]):
                            mapping[input_name].append(region)

        self.target_mapping[targets] = input_vars, mapping
        return input_vars, mapping

    def infer(self, shape: Dict[str, List[arith.ConstIntBound]], rstep: Dict[str, int]={}, targets=None):
        input_vars, mapping = self.construct_dependency_target(tuple(shape.keys()))
        ana = arith.Analyzer()
        results = {}
        for vars, bounds in zip(input_vars, shape.values()):
            for var, bound in zip(vars, bounds):
                ana.update(var, bound, True)
        for ax in self.reduce_axes:
            if ax.var.name in rstep:
                bound = arith.ConstIntBound(int(ax.dom.min), int(ax.dom.min + min(ax.dom.extent, rstep[ax.var.name]) - 1))
            else:
                bound = arith.ConstIntBound(int(ax.dom.min), int(ax.dom.min + ax.dom.extent - 1))
            ana.update(ax.var, bound, True)

        for name, regions in mapping.items():
            if targets is not None and name not in targets:
                continue
            for region in regions:
                bound = [ana.const_int_bound(indice) for indice in region]
                if name in results: # simply merge two bounds
                    bound = [_merge_two_bounds(x, y) for x, y in zip(results[name], bound)]
                results[name] = bound

        for name, bounds in results.items():
            results[name] = [c.max_value - c.min_value + 1 for c in bounds]
        return results

    def get_input_exprs(self, output_exprs):
        input_vars, mapping = self.construct_dependency_target(tuple(output_exprs.keys()))
        ana = arith.Analyzer()
        for ax in self.reduce_axes:
            ana.bind(ax.var, 0)
        vmap = {}
        for vars, exprs in zip(input_vars, output_exprs.values()):
            for var, expr in zip(vars, exprs):
                vmap[var] = expr
        result = {}

        for name, regions in mapping.items():
            region = regions[0]
            result[name] = [ana.simplify(tir.stmt_functor.substitute(index, vmap)) for index in region]
        return result

def region_exist_in_list(a, list) -> bool:
    def expr_is_same(a, b) -> bool:
        if isinstance(a, tir.IntImm) and isinstance(b, tir.IntImm):
            return a.value == b.value
        return a.same_as(b)

    def region_is_same(a, b) -> bool:
        for indice_a, indice_b in zip(a, b):
            if not expr_is_same(indice_a, indice_b):
                return False
        return True
    return any([region_is_same(a, x) for x in list])

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

def _extract_dependent_region(op: te.ComputeOp) -> Dict[str, List[tir.PrimExpr]]:
    dependent_region = {t.name: [] for t in op.input_tensors}

    def fvisit(x):
        if not isinstance(x, tir.ProducerLoad): return
        if x.producer.name not in dependent_region: return
        index = []
        for indice, shape_limit in zip(x.indices, x.producer.shape):
            expr = walk_indice(indice)
            if expr is None:
                expr = te.var("undefined") % shape_limit
            index.append(expr)
        if not region_exist_in_list(index, dependent_region[x.producer.name]):
            dependent_region[x.producer.name].append(index)

    for expr in op.body:
        tir.stmt_functor.post_order_visit(expr, fvisit=fvisit)
    return dependent_region


def get_analyzer_by_te(args : List[te.Tensor]) -> InputShapeInference:
    deps = [Statement(op) for op in get_compute_ops(args)]

    return InputShapeInference(deps)

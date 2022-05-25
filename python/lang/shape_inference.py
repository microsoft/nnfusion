from typing import Iterable, List, Dict
from collections import OrderedDict
from tvm import te, tir, arith
import copy
import numpy as np
from .einstein_v2 import parse_to_ast

class Statement():
    def __init__(self, output: str, dependent_region: dict, var_map: OrderedDict, range_map: OrderedDict):
        self.output = output
        self.dependent_region = dependent_region
        self.var_map = var_map
        self.range_map = range_map

def _merge_two_bounds(x: arith.ConstIntBound, y: arith.ConstIntBound):
    return arith.ConstIntBound(min(x.min_value, y.min_value), max(x.max_value, y.max_value))

class InputShapeInference():
    def __init__(self, deps: List[Statement]):
        self.deps = deps

    def _infer(self, shape: Dict[str, List[arith.ConstIntBound]], rstep: Dict[str, int]):
        shape = shape.copy()
        ana = arith.Analyzer()
        for dep in reversed(self.deps):
            for var, bound in zip(dep.var_map.values(), shape[dep.output]):
                ana.update(var, bound)
            for var, bound in dep.range_map.items():
                if var.name in rstep:
                    bound = arith.ConstIntBound(0, min(bound.max_value, rstep[var.name] - 1))
                ana.update(var, bound)
            for name, regions in dep.dependent_region.items():
                assert(len(regions) == 1) # TODO: merge if there is multiple region for one input
                for region in regions:
                    bounds = [ana.const_int_bound(index) for index in region]
                if name in shape: # simply merge two bounds
                    bounds = [_merge_two_bounds(x, y) for x, y in zip(shape[name], bounds)]
                shape[name] = bounds

        for name, bounds in shape.items():
            shape[name] = [c.max_value - c.min_value + 1 for c in bounds]
            assert(max(shape[name]) < 1e8) # checking
        return shape

    def infer(self, shape, rstep: Dict[str, int] = {}):
        if isinstance(shape, (list, tuple)):
            shape = {"output0" : [arith.ConstIntBound(0, val - 1) for val in shape]}
        shape = self._infer(shape, rstep)
        shape = dict(filter(lambda x: x[0].startswith("input"), shape.items()))
        return shape

    def get_reduction_inputs(self):
        # see what inputs are required in reductions stage.
        result = []
        for dep in self.deps:
            if len(dep.range_map) > 0 or dep.output in result:
                for name in dep.dependent_region:
                    result.append(name)
        return result

def get_analyzer(expr: str, input_dict: dict, extra_outputs: Iterable=[]) -> InputShapeInference:
    statements = [s_.strip() for s_ in expr.split(';')]
    inputs = copy.deepcopy(input_dict)
    output_dict = {}
    ast_seq = []
    for s in statements:
        if not s:
            continue
        ast = parse_to_ast(s, inputs)
        k = ast['props']['output_name']
        ast_outputs_dict = {
            k: {
                'shape': [x['range'] for x in ast['props']['data_axes']],
                'dtype': ast['root']._dtype
            }
        }
        inputs[k] = ast_outputs_dict[k]
        if k in extra_outputs:
            output_dict[k] = ast_outputs_dict[k]
        ast_seq.append(ast)
    # Also include the last output
    if k not in extra_outputs:
        output_dict[k] = ast_outputs_dict[k]

    deps = [_build_dependency(ast) for ast in ast_seq]

    return InputShapeInference(deps)


def _build_dependency(ast):
    props = ast['props']
    op = ast['root']
    output = props['output_name']
    data_axes = props['data_axes']
    reduce_axes = props['reduce_axes']

    var_map = OrderedDict()
    range_map = OrderedDict()
    for ax in data_axes:
        ax_name = ax['name']
        var_map[ax_name] = te.var(ax_name)
    for ax in reduce_axes:
        ax_name = ax['name']
        var_map[ax_name] = te.var(ax_name)
        range_map[var_map[ax_name]] = arith.ConstIntBound(0, ax['range'] - 1)
    dependent_region = {}

    def traverse(op):
        if op._op == 'get_item':
            nonlocal dependent_region
            tensor_name = op._value['tensor']._value
            if tensor_name not in dependent_region:
                dependent_region[tensor_name] = []
            index = []
            for ax in op._value['index']:
                index.append(_get_index_expr(ax, var_map))
            dependent_region[tensor_name].append(index)

        elif op._op in ['op', 'call', 'cast']:
            for in_op in op._value['inputs']:
                traverse(in_op)
        elif op._op == 'when':
            for in_op in op._value['if']:
                traverse(in_op)
            traverse(op._value['true'])
            traverse(op._value['false'])
        elif op._op in ['axis', 'const']:
            pass
        else:
            raise Exception('Unhandled node type in traverse(): %s' % op._op)

    traverse(op)
    return Statement(output, dependent_region, var_map, range_map)


def _get_index_expr(op, var_map):
    if op._op == 'const':
        return tir.const(int(op._value), 'int32')
    elif op._op == 'axis':
        return var_map[op._value]
    elif op._op == 'op':
        operator = op._value['name']
        args = []
        for in_op in op._value['inputs']:
            args.append(_get_index_expr(in_op, var_map))
        if len(args) == 2:
            expr = eval('args[0] {} args[1]'.format(operator))
        elif len(args) == 1:
            expr = eval('{} args[0]'.format(operator))
        else:
            raise Exception('Unhandled operator in _get_index_expr(): %s' % operator)
        return expr
    else:
        raise Exception('Unhandled node type in _get_index_expr(): %s' % op._op)

def get_analyzer_by_ir(antares_ir: str) -> InputShapeInference:
    antares_ir = antares_ir.strip()
    assert antares_ir.startswith(
        '- '
    ), "The computing expression doesn't start with proper prefix: - ..."

    antares_ir = antares_ir[2:]
    antares_ir = antares_ir.replace("einstein_v2", "get_analyzer")
    result = eval(antares_ir, globals(), locals())
    return result

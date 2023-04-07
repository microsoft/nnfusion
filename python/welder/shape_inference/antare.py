import copy
from collections import OrderedDict
from typing import Iterable, List

from tvm import arith, te, tir

from ..lang.einstein_v2 import parse_to_ast
from .common import InputShapeInference, Statement


def get_analyzer(expr: str, input_dict: dict, extra_outputs: Iterable=[]) -> InputShapeInference:
    statements = [s_.strip() for s_ in expr.split(';')]
    inputs = copy.deepcopy(input_dict)
    output_dict = {}
    shape_dict = {k : v['shape'] for k, v in input_dict.items()}
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
        shape_dict[k] = ast_outputs_dict[k]['shape']
        inputs[k] = ast_outputs_dict[k]
        if k in extra_outputs:
            output_dict[k] = ast_outputs_dict[k]
        ast_seq.append(ast)
    # Also include the last output
    if k not in extra_outputs:
        output_dict[k] = ast_outputs_dict[k]

    deps = [_build_dependency(ast, shape_dict) for ast in ast_seq]

    return InputShapeInference(deps)


def _build_dependency(ast, shape_dict):
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
            for ax, shape_limit in zip(op._value['index'], shape_dict[tensor_name]):
                expr = _get_index_expr(ax, var_map)
                if expr is None: # set to shape limit
                    expr = te.var("undefined") % shape_limit
                index.append(expr)
                traverse(ax)
            dependent_region[tensor_name].append(index)

        elif op._op in ['op', 'call', 'cast']:
            for in_op in op._value['inputs']:
                traverse(in_op)
        elif op._op == 'when':
            for in_op in op._value['if']:
                traverse(in_op)
            traverse(op._value['true'])
            traverse(op._value['false'])
        elif op._op in ['axis', 'const', 'axis_range']:
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
    elif op._op in ["call", "cast", "when", "get_item"]:
        return None
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

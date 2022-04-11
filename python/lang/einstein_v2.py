# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
import re
import copy
import json
import numpy as np

# Tensor name: the first charactor must be lower case letter, and the following charactors must be within [a-zA-Z_]
# Axis name: the first charactor must be upper case letter, and the following charactors must be within [a-zA-Z]

ast_props = None
explicit_range = None


class OpTensor:
    @staticmethod
    def parse(other):
        if isinstance(other, OpTensor):
            return other
        if isinstance(other, int):
            return OpTensor('const', other, 'int32', 0)
        if isinstance(other, float):
            return OpTensor('const', other, 'float32', 0)
        raise Exception("Unrecognized const node type: %s" % type(other))

    def filter_flop(self, other):
        if self._op == 'get_item' or other._op == 'get_item':
            return 1
        return 0

    def dtype(self):
        return self._dtype

    def __init__(self, _op, _value, _dtype, _flopbase=0):
        self._op = _op
        self._value = _value
        self._dtype = _dtype
        self._flopbase = _flopbase

    def __repr__(self):
        return 'OpTensor{"%s", "%s", "%s"}' % (self._op, self._value,
                                               self._dtype)

    def __getitem__(self, key):
        if self._op != 'tensor':
            raise Exception(
                "The instance to access its dim values must be a tensor array."
            )
        key = list(key if isinstance(key, tuple) else (key, ))
        _flopbase = self._flopbase
        for i in range(len(key)):
            key[i] = OpTensor.parse(key[i])
            it = key[i]
            _flopbase += it._flopbase
            if it._op == 'axis' and explicit_range[it._value] is None:
                explicit_range[it._value] = ast_props['input_dict'][
                    self._value]['shape'][i]
        return OpTensor('get_item', {
            "tensor": self,
            "index": key
        }, self._dtype, _flopbase)

    # Calculation Ops
    def __mul__(self, other):
        other = OpTensor.parse(other)
        if other._op == 'const' and other._value == 1:
            return self
        if self._op == 'const' and self._value == 1:
            return other
        return OpTensor(
            'op', {
                "name": "*",
                "inputs": [self, other]
            }, self._dtype,
            self._flopbase + other._flopbase + self.filter_flop(other))

    def __rmul__(self, other):
        other = OpTensor.parse(other)
        return other.__mul__(self)

    def __truediv__(self, other):
        other = OpTensor.parse(other)
        op_name = '//' if self._dtype == 'int32' and other._dtype == 'int32' else '/'
        if other._op == 'const' and other._value == 1:
            return self
        if other._op == 'const' and self._op == 'axis':
            assert self._value in explicit_range and explicit_range[
                self._value] is not None
            if op_name == '//' and explicit_range[self._value] < other._value:
                return OpTensor.parse(int(0))
        return OpTensor(
            'op', {
                "name": op_name,
                "inputs": [self, other]
            }, self._dtype,
            self._flopbase + other._flopbase + self.filter_flop(other))

    def __rtruediv__(self, other):
        other = OpTensor.parse(other)
        return other.__truediv__(self)

    def __floordiv__(self, other):
        other = OpTensor.parse(other)
        return self.__truediv__(other)

    def __rfloordiv__(self, other):
        other = OpTensor.parse(other)
        return other.__floordiv__(self)

    def __mod__(self, other):
        other = OpTensor.parse(other)
        if other._op == 'const':
            assert other._dtype == 'int32'
            if other._value == 1:
                return OpTensor.parse(int(0))
            if self._op == 'axis':
                assert self._value in explicit_range and explicit_range[
                    self._value] is not None
                if explicit_range[self._value] <= other._value:
                    return self
        return OpTensor(
            'op', {
                "name": "%",
                "inputs": [self, other]
            }, self._dtype,
            self._flopbase + other._flopbase + self.filter_flop(other))

    def __add__(self, other):
        other = OpTensor.parse(other)
        if other._op == 'const' and other._value == 0:
            return self
        if self._op == 'const' and self._value == 0:
            return other
        return OpTensor(
            'op', {
                "name": "+",
                "inputs": [self, other]
            }, self._dtype,
            self._flopbase + other._flopbase + self.filter_flop(other))

    def __radd__(self, other):
        other = OpTensor.parse(other)
        return other.__add__(self)

    def __sub__(self, other):
        other = OpTensor.parse(other)
        if other._op == 'const' and other._value == 0:
            return self
        return OpTensor(
            'op', {
                "name": "-",
                "inputs": [self, other]
            }, self._dtype,
            self._flopbase + other._flopbase + self.filter_flop(other))

    def __rsub__(self, other):
        other = OpTensor.parse(other)
        return other.__sub__(self)

    def __neg__(self):
        return OpTensor.parse(0).cast(self._dtype).__sub__(self)

    # Relation Ops
    def __lt__(self, other):
        other = OpTensor.parse(other)
        return OpTensor('op', {
            "name": "<",
            "inputs": [self, other]
        }, 'int8', self._flopbase + other._flopbase)

    def __le__(self, other):
        other = OpTensor.parse(other)
        return OpTensor('op', {
            "name": "<=",
            "inputs": [self, other]
        }, 'int8', self._flopbase + other._flopbase)

    def __gt__(self, other):
        other = OpTensor.parse(other)
        return OpTensor('op', {
            "name": "<",
            "inputs": [other, self]
        }, 'int8', self._flopbase + other._flopbase)

    def __ge__(self, other):
        other = OpTensor.parse(other)
        return OpTensor('op', {
            "name": "<=",
            "inputs": [other, self]
        }, 'int8', self._flopbase + other._flopbase)

    def __eq__(self, other):
        other = OpTensor.parse(other)
        return OpTensor('op', {
            "name": "==",
            "inputs": [self, other]
        }, 'int8', self._flopbase + other._flopbase)

    def __ne__(self, other):
        other = OpTensor.parse(other)
        return OpTensor('op', {
            "name": "!=",
            "inputs": [self, other]
        }, 'int8', self._flopbase + other._flopbase)

    def __and__(self, other):
        other = OpTensor.parse(other)
        return OpTensor('op', {
            "name": "&",
            "inputs": [self, other]
        }, 'int8', self._flopbase + other._flopbase)

    def __or__(self, other):
        other = OpTensor.parse(other)
        return OpTensor('op', {
            "name": "|",
            "inputs": [self, other]
        }, 'int8', self._flopbase + other._flopbase)

    def __invert__(self):
        return OpTensor('op', {
            "name": "~",
            "inputs": [self]
        }, 'int8', self._flopbase)

    # Special Ops
    def cast(self, dtype):
        return OpTensor('cast', {
            "name": dtype,
            "inputs": [self]
        }, dtype, self._flopbase)

    def call(self, func_name, others=None, dtype=None):
        if others is None:
            others = []
        _flopbase = self._flopbase + self.filter_flop(self)
        for i in range(len(others)):
            others[i] = OpTensor.parse(others[i])
            _flopbase += others[i]._flopbase
        if dtype is None:
            dtype = self._dtype
        return OpTensor('call', {
            "name": func_name,
            "inputs": [self] + others
        }, dtype, _flopbase)

    def when(self, conditions, other):
        other = OpTensor.parse(other)
        assert self._dtype == other._dtype or '@' in self._dtype or '@' in other._dtype, "Conditional true and false values must have same datatype (%s v.s. %s)" % (
            self._dtype, other._dtype)
        conditions = conditions if isinstance(conditions,
                                              list) else [conditions]
        for cond in conditions:
            assert (cond._dtype == 'int8')
        return OpTensor('when', {
            "if": conditions,
            "true": self,
            "false": other
        }, self._dtype, max(self._flopbase, other._flopbase))


def parse_to_ast(expr, input_dict={}):
    expr = expr.strip().replace('`', '"')
    if re.search('\[ *\]', expr):
        expr = re.sub('\[ *\]', '[0]', expr)
        if expr.rfind('where') == -1:
            expr += ' where Scaler in 1'
        else:
            expr += ', Scaler in 1'
    at_index = expr.rfind(' where ')
    if at_index != -1:
        range_desc = expr[at_index + len(' where '):]
        expr = expr[:at_index]
    else:
        range_desc = ''

    # Parse compute axes & init axis nodes
    global explicit_range
    explicit_range = {}
    for i in range(1, len(expr)):
        if expr[i].isupper() and (not expr[i - 1].isalpha()) and (
                not expr[i - 1].isdigit()) and (expr[i - 1] != '_'):
            for j in range(i, len(expr) + 1):
                if j == len(expr) or (not expr[j].isalpha()
                                      and not expr[j].isdigit()):
                    ax_name = expr[i:j]
                    break
            if ax_name not in explicit_range:
                explicit_range[ax_name] = None
    exec("_id = OpTensor('axis', '_id', 'int32')")
    for k in explicit_range:
        exec("%s = OpTensor('axis', k, 'int32')" % k)

    # Parse where clause
    for x in range_desc.split(','):
        x = x.strip()
        if not x:
            continue
        k, v = x.split(' in ')
        explicit_range[k.strip()] = int(v.strip())

    # Parse compute set-op, get lval & rval
    props = {
        'data_axes': [],
        'reduce_axes': [],
        'input_dict': copy.deepcopy(input_dict),
        'output_name': None,
        'reduce_type': None,
        'flopbase': None
    }
    global ast_props
    ast_props = props

    at_index = expr.find('=')
    if expr[at_index - 1] != ' ':
        if expr[at_index - 1] in ('<', '>', '+'):
            props['reduce_type'] = expr[at_index - 1]
            lval = expr[:at_index - 1].strip()
        else:
            blank_index = expr.find(' ', 0, at_index)
            assert blank_index > 0, "Illegal reduce naming in equation near: `L-value <reduce_type>=`"
            props['reduce_type'] = expr[blank_index + 1:at_index]
            lval = expr[:blank_index].strip()
    else:
        lval = expr[:at_index].strip()
    if expr[at_index + 1] == '!':
        assert (props['reduce_type'] is not None)
        rval = expr[at_index + 2:].strip()
    else:
        rval = expr[at_index + 1:].strip()

    # Distinguish data/reduce axes according to lval
    for x in lval[lval.index('[') + 1:lval.rindex(']')].split(','):
        x = x.strip()
        if x == '0':
            x = 'Scaler'
        props['data_axes'].append(x)
    for x in explicit_range:
        if x not in props['data_axes'] and x != 'Scaler':
            props['reduce_axes'].append(x)

    for input_name in input_dict:
        if not input_name[0].islower():
            raise Exception(
                "Tensor variable name must start with lower case letter: %s" %
                input_name)
        exec('%s = OpTensor("tensor", input_name, "%s")' %
             (input_name, input_dict[input_name]["dtype"]))

    # Build ast according to rval & fill uncertain axis range
    _root = eval(rval)
    for x in explicit_range:
        if explicit_range[x] is None:
            raise Exception(
                "The range of axis `%s` is undeterminzed, please use `where` clause to set the range explicitly."
                % x)

    # Collect output inferences & compute flopbase
    props['flopbase'] = max(
        1,
        _root._flopbase if props['reduce_type'] is None else _root._flopbase +
        1)

    props['data_axes'] = [{
        'name': x,
        'range': explicit_range[x]
    } for x in props['data_axes']]
    props['reduce_axes'] = [{
        'name': x,
        'range': explicit_range[x]
    } for x in props['reduce_axes']]

    output_name = lval[:lval.index('[')].strip()
    props['output_name'] = output_name
    return {'props': props, 'root': _root}


def const(other):
    return OpTensor.parse(other)


def warp_axis(ax_name):
    assert (ax_name[0].isupper() or ax_name == '_id')
    return ax_name


def emit_antares_ir(ast):
    def _emit(node):
        if node._op == 'const':
            return 'const(%s)' % node._value
        elif node._op == 'axis':
            if hasattr(node, '_func'):
                return node._func(node._value)
            return node._value
        elif node._op == 'op':
            if len(node._value['inputs']) == 2:
                return '(%s %s %s)' % (_emit(
                    node._value['inputs'][0]), node._value['name'],
                                       _emit(node._value['inputs'][1]))
            raise
        elif node._op == 'get_item':
            return '%s[%s]' % (node._value['tensor']._value, ', '.join(
                [_emit(x) for x in node._value['index']]))
        elif node._op == 'call':
            if len(node._value['inputs']) == 1:
                return '(%s).call(`%s`, dtype=`%s`)' % (_emit(
                    node._value['inputs'][0]), node._value['name'],
                                                        node._dtype)
            return '(%s).call(`%s`, [%s], dtype=`%s`)' % (_emit(
                node._value['inputs'][0]), node._value['name'], ', '.join(
                    [_emit(x)
                     for x in node._value['inputs'][1:]]), node._dtype)
        elif node._op == 'when':
            if len(node._value['if']) == 0:
                return '(%s)' % _emit(node._value['true'])
            return '(%s).when([%s], %s)' % (_emit(
                node._value['true']), ', '.join(
                    [_emit(x)
                     for x in node._value['if']]), _emit(node._value['false']))
        elif node._op == 'cast':
            return '(%s).cast(`%s`)' % (_emit(
                node._value['inputs'][0]), node._dtype)
        else:
            raise Exception(
                "Emit Antares IR: Unhanled reverse-emit op type: %s" %
                node._op)

    lval = '%s[%s]' % (ast['props']['output_name'], ', '.join(
        [x['name'] for x in ast['props']['data_axes']]))
    comp_type = '%s=%s' % (ast['props']['reduce_type']
                           if ast['props']['reduce_type'] else '',
                           '!' if ast['props']['reduce_type'] else '')
    return '%s %s %s where %s;' % (lval, comp_type, _emit(
        ast['root']), ', '.join([
            '%s in %d' % (x['name'], x['range'])
            for x in ast['props']['data_axes'] + ast['props']['reduce_axes']
        ]))


def emit_tvm_body(node, props):
    if node._op == 'const':
        return 'tir.const(%s, dtype="%s")' % (node._value, node._dtype)
    elif node._op == 'get_item':
        tensor = node._value['tensor']
        index = node._value['index']
        _str = tensor._value + '['
        if len(index) > 0:
            for i, it in enumerate(index):
                _str += emit_tvm_body(it, props) + ', '
            _str = _str[:-2] + ']'
        return _str
    elif node._op == 'axis':
        axis_name = warp_axis(node._value)
        if hasattr(node, '_func'):
            axis_name = node._func(axis_name)
        return axis_name
    elif node._op == 'op':
        op_name = node._value["name"]
        op_input_size = len(node._value["inputs"])
        if op_name in ('&', '|', '~'):
            if op_name == '&':
                return 'te.all(' + emit_tvm_body(
                    node._value["inputs"][0],
                    props) + '.astype("bool"), ' + emit_tvm_body(
                        node._value["inputs"][1], props) + '.astype("bool"))'
            elif op_name == '|':
                return 'te.any(' + emit_tvm_body(
                    node._value["inputs"][0],
                    props) + '.astype("bool"), ' + emit_tvm_body(
                        node._value["inputs"][1], props) + '.astype("bool"))'
            else:
                return '(' + emit_tvm_body(node._value["inputs"][0],
                                           props) + ' == 0)'
        elif op_input_size == 2:
            return '(' + emit_tvm_body(
                node._value["inputs"][0],
                props) + ' ' + op_name + ' ' + emit_tvm_body(
                    node._value["inputs"][1], props) + ')'
        elif op_input_size == 1:
            return '(' + op_name + emit_tvm_body(node._value["inputs"][0],
                                                 props) + ')'
        else:
            raise Exception('Unrecognized op type: %s[%d]' %
                            (op_name, op_input_size))
    elif node._op == 'cast':
        return '%s.astype(cast_dtype("%s"))' % (emit_tvm_body(
            node._value["inputs"][0], props), node._value['name'])
    elif node._op == 'call':
        return 'tir.call_pure_extern(cast_dtype("%s"), "%s", %s)' % (
            node._dtype, node._value['name'], ', '.join(
                [emit_tvm_body(x, props) for x in node._value["inputs"]]))
    elif node._op == 'when':
        all_conds = [emit_tvm_body(cond, props) for cond in node._value['if']]
        return 'tir.if_then_else(te.all(' + ', '.join(
            all_conds) + '), t=' + emit_tvm_body(
                node._value['true'], props) + ', f=' + emit_tvm_body(
                    node._value['false'], props) + ')'
    else:
        raise Exception('Unrecognized node type: %s' % node._op)


def walk_in_ast(node, func, args, parent, attr_id):
    def _walk(node, parent, attr_id):
        updated_node = func(node, *args)
        if updated_node is not None:
            if isinstance(updated_node, str) and updated_node == '':
                return
            updated_node = copy.deepcopy(updated_node)
            if isinstance(parent, OpTensor):
                setattr(parent, attr_id, updated_node)
            else:
                parent[attr_id] = updated_node
            return
        if node._op == 'get_item':
            for i, ch in enumerate(node._value['index']):
                _walk(ch, node._value['index'], i)
        elif node._op in ['op', 'call', 'cast']:
            for i, ch in enumerate(node._value['inputs']):
                _walk(ch, node._value['inputs'], i)
        elif node._op == 'when':
            for i, ch in enumerate(node._value['if']):
                _walk(ch, node._value['if'], i)
            _walk(node._value['true'], node._value, 'true')
            _walk(node._value['false'], node._value, 'false')
        elif node._op in ['axis', 'const']:
            pass
        else:
            raise Exception('Unhandled node type in walk_in_ast(): %s' %
                            node._op)

    _walk(node, parent, attr_id)


def apply_fusion(ast, top_ast):
    def _replace_axis(node, replace_maps):
        if node._op == 'axis' and node._value['name'] in replace_maps:
            return replace_maps[node._value['name']]
        return None

    def _replace_tensor(node):
        if node._op == 'get_item':
            tensor_name = node._value['tensor']._value['name']
            if tensor_name not in top_ast:
                return None
            sub_ast = copy.deepcopy(top_ast[tensor_name])
            replace_maps = {}
            for i in range(len(node._value['index'])):
                replace_maps[sub_ast['props']['data_axes'][i]
                             ['name']] = node._value['index'][i]
            walk_in_ast(sub_ast['root'], _replace_axis, [replace_maps],
                        sub_ast, 'root')
            return sub_ast['root']
        return None

    walk_in_ast(ast['root'], _replace_tensor, [], ast, 'root')
    return ast


def emit_tvm_ir_v2(exprss, input_dict, extra_outputs):
    statements = [s_.strip() for s_ in exprss.split(';')]
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
                "shape": [x['range'] for x in ast['props']['data_axes']],
                "dtype": ast['root']._dtype
            }
        }
        inputs[k] = ast_outputs_dict[k]
        if k in extra_outputs:
            output_dict[k] = ast_outputs_dict[k]
        ast_seq.append(ast)

    # Also include the last output
    if k not in extra_outputs:
        output_dict[k] = ast_outputs_dict[k]

    # Registry Global Argument Properties
    arg_props = {'_in': [], '_out': []}
    for k in input_dict:
        prop = copy.deepcopy(input_dict[k])
        prop['name'] = k
        arg_props['_in'].append(prop)
    for k in output_dict:
        prop = copy.deepcopy(output_dict[k])
        prop['name'] = k
        arg_props['_out'].append(prop)
    arg_props['_in'].sort(key=lambda x: x['name'])
    arg_props['_out'].sort(key=lambda x: x['name'])

    # import importlib
    # passes = os.listdir('lang/pass')
    # passes.sort()
    # for pas in passes:
    #     if pas.endswith('.py'):
    #         pass_stage = importlib.import_module('lang.pass.%s' % pas[:-3])
    #         pass_stage.run_pass_v2(ast_seq, input_dict, output_dict)

    # Generate LL_IR body for ast_seq
    def emit_input_body(input_dict):
        # input_body = '_id = input("_id", [1], dtype="int32")[0]; '
        input_body = str()
        for key in input_dict:
            input_info = input_dict[key]
            input_body += '%s = input("%s", %s, dtype="%s"); ' % (
                key, key, input_info['shape'], input_info['dtype'])
        return input_body

    def emit_reduce_body(ast):
        reduce_body, reduce_set = '', []
        props = ast['props']
        if props['reduce_axes']:
            for x in props['reduce_axes']:
                axis_name = warp_axis(x['name'])
                reduce_set.append(axis_name)
                reduce_body += '%s = loop(%d); ' % (axis_name, x['range'])
            reduce_maps = {'+': 'te.sum', '>': 'te.max', '<': 'te.min'}
            if props['reduce_type'] in reduce_maps:
                reduce_func = reduce_maps[props['reduce_type']]
            else:
                spec_idx = props['reduce_type'].find('(')
                if spec_idx >= 0:
                    reduce_func = 'common_reduce("%s", %s)' % (
                        props['reduce_type'][:spec_idx],
                        props['reduce_type'][spec_idx:])
                else:
                    reduce_func = 'common_reduce("%s")' % props['reduce_type']
            reduce_pattern = '%s(' % reduce_func + '%s' + ', axis=[%s])' % ', '.join(
                reduce_set)
        else:
            reduce_pattern = '%s'
        return reduce_body, reduce_pattern

    def emit_output_body(ast, reduce_pattern):
        root, props = ast['root'], ast['props']
        output_shape = [x['range'] for x in props['data_axes']]
        output_name = props['output_name']
        all_axis_range = np.product(output_shape) * np.product(
            [x['range'] for x in props['reduce_axes']])
        output_begin = '%s = output(shape=%s, flops=(%d * %d), func=lambda %s: ' % (
            output_name, output_shape, props['flopbase'], all_axis_range,
            ', '.join([warp_axis(x['name']) for x in props['data_axes']]))
        basic_body = emit_tvm_body(root, props)
        output_end = ', dtype="%s", tag="%s", name="%s", final_output=%s); ' % (
            root._dtype, '', output_name, output_name in output_dict)
        return output_begin + reduce_pattern % basic_body + output_end

    ll_irs = [emit_input_body(input_dict)]
    for ast in ast_seq:
        loops_def, pattern = emit_reduce_body(ast)
        ll_irs.append(loops_def + emit_output_body(ast, pattern))
    return '\n'.join(ll_irs)


def emit_tvm_ir(exprss, input_dict, extra_outputs):
    return emit_tvm_ir_v2(exprss, input_dict, extra_outputs)

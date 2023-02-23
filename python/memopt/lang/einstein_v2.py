# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import copy
import re

import numpy as np

# Tensor name: the first charactor must be lower case letter, and the following charactors must be within [a-zA-Z_]
# Axis name: the first charactor must be upper case letter, and the following charactors must be within [a-zA-Z]

ast_props = None
explicit_range = None


class OpTensor:
    @staticmethod
    def parse(other, output_dtype=None):
        if isinstance(other, OpTensor):
          return other.cast(output_dtype)
        if output_dtype is not None:
          return OpTensor('const', other, output_dtype)
        if isinstance(other, int):
          return OpTensor('const', other, 'int32')
        if isinstance(other, float):
          return OpTensor('const', other, 'float32')
        raise Exception("Unrecognized const node type: %s" % type(other))

    @staticmethod
    def merge_dtype(first, second):
        dtypes = (first._dtype, second._dtype)
        ordered_dtypes = ['float64', 'float32', 'int32', 'int16', 'int8']
        for _dtype in ordered_dtypes:
          if _dtype in dtypes:
            return _dtype
        return first._dtype

    def dtype(self):
        return self._dtype

    def val(self):
        assert self._op == 'axis', "Only axis op can support value fetch for its range."
        return OpTensor('axis_range', self._value, 'int32')

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
        output_dtype = OpTensor.merge_dtype(self, other)
        if other._op == 'const' and other._value == 1:
            return self.cast(output_dtype)
        if self._op == 'const' and self._value == 1:
            return other.cast(output_dtype)
        return OpTensor('op', {"name": "*", "inputs": [self.cast(output_dtype), other.cast(output_dtype)]}, output_dtype)

    def __rmul__(self, other):
        other = OpTensor.parse(other)
        return other.__mul__(self)

    def __truediv__(self, other):
        other = OpTensor.parse(other)
        op_name = '//' if re.match(r'^int[0-9]+$', self._dtype) and re.match(r'^int[0-9]+$', other._dtype) else '/'
        output_dtype = OpTensor.merge_dtype(self, other)
        if other._op == 'const' and other._value == 1:
            return self.cast(output_dtype)
        if other._op == 'const' and self._op == 'axis':
            if self._value in explicit_range and explicit_range[self._value] is not None:
                if op_name == '//' and explicit_range[self._value] < other._value:
                    return OpTensor.parse(0, output_dtype)
        result = OpTensor('op', {"name": op_name, "inputs": [self, other]}, output_dtype)
        if 'float' in self._dtype and 'int' in other._dtype:
            result = result.cast(self._dtype)
        return result

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
                return OpTensor.parse(0, self._dtype)
            if self._op == 'axis':
                if (explicit_range.get(self._value) or other._value + 1) <= other._value:
                    return self
        return OpTensor('op', {"name": "%", "inputs": [self, other]}, self._dtype)

    def __add__(self, other):
        other = OpTensor.parse(other)
        output_dtype = OpTensor.merge_dtype(self, other)
        if other._op == 'const' and other._value == 0:
            return self.cast(output_dtype)
        if self._op == 'const' and self._value == 0:
            return other.cast(output_dtype)
        return OpTensor('op', {"name": "+", "inputs": [self.cast(output_dtype), other.cast(output_dtype)]}, output_dtype)

    def __radd__(self, other):
        other = OpTensor.parse(other)
        return other.__add__(self)

    def __sub__(self, other):
        other = OpTensor.parse(other)
        output_dtype = OpTensor.merge_dtype(self, other)
        if other._op == 'const' and other._value == 0:
            return self.cast(output_dtype)
        return OpTensor('op', {"name": "-", "inputs": [self.cast(output_dtype), other.cast(output_dtype)]}, output_dtype)

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
    def cast(self, output_dtype):
        if output_dtype is None or self._dtype == output_dtype:
          return self
        return OpTensor('cast', {"inputs": [self]}, output_dtype)

    def call(self, func_name, others=None, output_dtype=None):
        if others is None:
            others = []
        for i in range(len(others)):
            others[i] = OpTensor.parse(others[i])
        if func_name == 'remainder' and len(others) == 0:
            return self - self.cast('int64' if self._dtype == 'float64' else 'int32')
        if func_name == 'floor' and len(others) == 0:
            return self.cast('int64' if self._dtype == 'float64' else 'int32')
        if func_name == 'ceil' and len(others) == 0:
            floor_op = self.cast('int64' if self._dtype == 'float64' else 'int32')
            return floor_op.when(self == floor_op, floor_op + const(1).cast(floor_op._dtype))
        if output_dtype is None:
            output_dtype = self._dtype
        return OpTensor('call', {"name": func_name, "inputs": [self] + others}, output_dtype)

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
            node._value["inputs"][0], props), node._dtype)
    elif node._op == 'call':
        f_map = {"max": "tir.Max", "min": "tir.Min", "exp": "tir.exp", "ceil": "tir.ceil", "erf" : "te.erf", "pow" : "te.power",
            "tanh": "te.tanh", "tan": "te.tan", "sqrt": "te.sqrt", "log": "te.log"}
        if node._value['name'] in f_map:
            return '%s(%s)' % (f_map[node._value['name']], ', '.join(
                    [emit_tvm_body(x, props) for x in node._value["inputs"]]))
        else:
            return 'tir.call_pure_extern(cast_dtype("%s"), "%s", %s)' % (
                node._dtype, node._value['name'], ', '.join(
                    [emit_tvm_body(x, props) for x in node._value["inputs"]]))
    elif node._op == 'when':
        all_conds = [emit_tvm_body(cond, props) for cond in node._value['if']]
        return 'tir.if_then_else(te.all(' + ', '.join(
            all_conds) + '), t=' + emit_tvm_body(
                node._value['true'], props) + ', f=' + emit_tvm_body(
                    node._value['false'], props) + ')'
    elif node._op == 'axis_range':
        for x in props['data_axes'] + props['reduce_axes']:
            if x['name'] == node._value:
                return 'tir.const(%s, dtype="%s")' % (x['range'], node._dtype)
        raise Exception('axes_range for %s is not found.' % node._value)
    else:
        raise Exception('Unrecognized node type: %s' % node._op)


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
                reduce_body += '%s = loop(%d, "%s"); ' % (axis_name, x['range'], axis_name)
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

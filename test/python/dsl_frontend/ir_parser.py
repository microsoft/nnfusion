# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
import re
import copy
import json

# Tensor name: the first charactor must be lower case letter, and the following charactors must be within [a-zA-Z_]
# Axis name: the first charactor must be upper case letter, and the following charactors must be within [a-zA-Z]

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
        ordered_dtypes = ['float64', 'float32', 'float16', 'int32', 'int16', 'int8']
        for _dtype in ordered_dtypes:
          if _dtype in dtypes:
            return _dtype
        return first._dtype

    def dtype(self):
        return self._dtype

    def alter(self, name):
        assert self._op == 'const', "Only constant value is allowed for function alter()."
        self._op = 'alter'
        explicit_range[name] = (name, int(self._value)) if self._dtype == 'int32' else (name, (self._value, self._dtype))
        self._alter_name = name
        return self

    def val(self, dim=None):
        if self._op == 'tensor':
          size = vamap_tensor[self._value][dim]
          if isinstance(size, str):
            return const(full_tensor_dict[self._value]['shape'][dim]).alter(size)
          return size
        if self._op == 'axis':
          return OpTensor('axis_range', self._value, 'int32')
        raise Exception("Only tensor/axis objects have val property, get: %s" % self._op)

    def __init__(self, _op, _value, _dtype):
        self._op = _op
        self._value = _value
        self._dtype = _dtype

    def __repr__(self):
        return 'OpTensor{"%s", "%s", "%s"}' % (self._op, self._value, self._dtype)

    def __getitem__(self, key):
        if self._op != 'tensor':
            raise Exception("The instance to access its dim values must be a tensor array.")
        key = list(key if isinstance(key, tuple) else (key, ))
        for i in range(len(key)):
          key[i] = OpTensor.parse(key[i])
          it = key[i]
          if it._op == 'get_item':
            os.environ['SPECIAL'] = '1'
          if it._op == 'axis' and explicit_range[it._value] is None:
            k = vamap_tensor[self._value][i]
            v = full_tensor_dict[self._value]['shape'][i]
            explicit_range[it._value] = (k, v)
        return OpTensor('get_item', {"tensor": self, "index": key}, self._dtype)

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
            const_value = (explicit_range.get(self._value) or (None,))[0]
            if op_name == '//' and isinstance(const_value, int) and const_value <= other._value:
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
                const_value = (explicit_range.get(self._value) or (None,))[0]
                if isinstance(const_value, int) and const_value <= other._value:
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

    def __rshift__(self, other):
        other = OpTensor.parse(other)
        return OpTensor('op', {"name": ">>", "inputs": [self, other]}, self._dtype)

    def __lshift__(self, other):
        other = OpTensor.parse(other)
        return OpTensor('op', {"name": "<<", "inputs": [self, other]}, self._dtype)

    def __neg__(self):
        return OpTensor.parse(0, self._dtype).__sub__(self)

    # Relation Ops
    def __lt__ (self, other):
        other = OpTensor.parse(other)
        return OpTensor('op', {"name": "<", "inputs": [self, other]}, 'int8')

    def __le__ (self, other):
        other = OpTensor.parse(other)
        return OpTensor('op', {"name": "<=", "inputs": [self, other]}, 'int8')

    def __gt__ (self, other):
        other = OpTensor.parse(other)
        return OpTensor('op', {"name": "<", "inputs": [other, self]}, 'int8')

    def __ge__ (self, other):
        other = OpTensor.parse(other)
        return OpTensor('op', {"name": "<=", "inputs": [other, self]}, 'int8')

    def __eq__ (self, other):
        other = OpTensor.parse(other)
        return OpTensor('op', {"name": "==", "inputs": [self, other]}, 'int8')

    def __ne__ (self, other):
        other = OpTensor.parse(other)
        return OpTensor('op', {"name": "!=", "inputs": [self, other]}, 'int8')

    def __and__(self, other):
        other = OpTensor.parse(other)
        return OpTensor('op', {"name": "&", "inputs": [self, other]}, 'int8')

    def __or__(self, other):
        other = OpTensor.parse(other)
        return OpTensor('op', {"name": "|", "inputs": [self, other]}, 'int8')

    def __invert__(self):
        return OpTensor('op', {"name": "~", "inputs": [self]}, 'int8')

    # Special Ops
    def cast(self, output_dtype):
        if output_dtype is None or self._dtype == output_dtype:
          return self
        return OpTensor('cast', {"inputs": [self]}, output_dtype)

    def up_cast(self):
        if self._dtype in ('int8', 'int16',):
          return self.cast('int32')
        if self._dtype in ('float16',):
          return self.cast('float32')
        return self

    def call(self, func_name, others=None, output_dtype=None):
        if others is None:
          others = []
        if isinstance(others, tuple):
          others = list(others)
        if not isinstance(others, list):
          others = [others]
        for i in range(len(others)):
          others[i] = OpTensor.parse(others[i])
        if func_name == 'sigmoid' and len(others) == 0:
          return 1 / (self.call('exp') + 1)
        if func_name == 'remainder' and len(others) == 0:
          return self - self.call('rfloor')
        if func_name in ('floor', 'ceil') and len(others) == 0:
          output = OpTensor('call', {"name": func_name, "inputs": [self.up_cast()]}, self._dtype)
          return output.cast('int64' if self._dtype == 'float64' else 'int32')
        if func_name in ('rfloor', 'rceil') and len(others) == 0:
          func_name = func_name[1:]
        if func_name in ('exp', 'sqrt', 'rsqrt', 'max', 'min', 'log', 'sin', 'cos', 'floor', 'ceil') and self._dtype == 'float16':
          func_name = f'fp16_{func_name}'
        if func_name in ('pow', 'abs', 'tan', 'tanh', 'normcdf', 'erf'):
          return OpTensor('call', {"name": func_name, "inputs": [OpTensor.parse(x).up_cast() for x in [self] + others]}, self.up_cast().dtype()).cast(self._dtype)
        if output_dtype is None:
          output_dtype = self._dtype
        return OpTensor('call', {"name": func_name, "inputs": [self] + others}, output_dtype)

    def when(self, conditions, other, merge_op='all'):
        other = OpTensor.parse(other)
        assert self._dtype == other._dtype or '@' in self._dtype or '@' in other._dtype, "Conditional true and false values must have same datatype (%s v.s. %s)" % (self._dtype, other._dtype)
        conditions = conditions if isinstance(conditions, list) else [conditions]
        for cond in conditions:
          assert cond._dtype.startswith('int'), 'Each condition in when statement must be integer type, get: %s' % cond._dtype
        return OpTensor('when', {"if": conditions, "true": self, "false": other, "merge_op": merge_op}, self._dtype)

def get_valid_axis(expr):
  valid_axis = set()
  for i, section in enumerate(expr.split('"')):
    if i % 2 == 1:
      continue
    word = re.search(r'\b[A-Z][a-zA-Z0-9]*', section)
    while word is not None:
      valid_axis.add(section[word.start():word.end()])
      section = section[word.end():]
      word = re.search(r'\b[A-Z][a-zA-Z0-9]*', section)
  return valid_axis


def detach_where_clause(expr):
  at_index = expr.rfind(' where ')
  if at_index != -1:
    range_desc = expr[at_index + len(' where '):]
    expr = expr[:at_index]
  else:
    range_desc = ''

  range_items = dict()
  for x in range_desc.split(','):
    x = x.strip()
    if not x:
      continue
    k, v = x.split(' in ')
    k = k.strip()
    if ':' in v:
      n, v = v.split(':')
      v = int(v.strip())
      range_items[k] = (n.strip(), v)
    else:
      v = int(v.strip())
      range_items[k] = (v, v)
  return expr, range_items


def parse_to_ast(expr):
  expr = expr.strip().replace('`', '"').replace('\'', '"').replace('=!', '=')

  # Construct explicit_range
  global explicit_range
  expr, explicit_range = detach_where_clause(expr)

  valid_axis = get_valid_axis(expr)
  for ax_name in valid_axis:
    if ax_name not in explicit_range:
      explicit_range[ax_name] = None

  # Enable special syntax
  set_index = expr.find('=.')
  if set_index >= 1:
    end_index = set_index + 2
    start_index = set_index - 1 if expr[set_index - 1] in ('+', '<', '>', '(', ')')  else set_index
    set_op, symbolic_output = expr[start_index:end_index], expr[:expr.index('[')].strip()
    if set_op == '=.':
      set_op = '__builtin_set'
    elif set_op == '+=.':
      set_op = '__builtin_add'
    elif set_op == '<=.':
      set_op = '__builtin_min'
    elif set_op == '>=.':
      set_op = '__builtin_max'
    elif set_op == '(=.':
      set_op = '__builtin_argmin'
    elif set_op == ')=.':
      set_op = '__builtin_argmax'
    else:
      raise Exception(f'Unimplemented set_op: `{set_op}`')
    expr = f'___{symbolic_output}[{", ".join(valid_axis)}] = {expr[:start_index].strip()}.call("{set_op}", [{expr[end_index:].strip()},])'

  # Handle scaler tensor
  if re.search('\[ *\]', expr):
    expr = re.sub('\[ *\]', '[0]', expr)
    i = 0
    while f'I{i}' in explicit_range:
      i += 1
    scale_ax_name = f'I{i}'
    explicit_range[scale_ax_name] = (1, 1)

  # Init axis nodes
  exec("_id = OpTensor('axis', '_id', 'int32')")
  for k in explicit_range:
    exec("%s = OpTensor('axis', k, 'int32')" % k)

  props = {'data_axes': [], 'reduce_axes': [], 'input_dict': None, 'output_name': None, 'reduce_type': None}

  # Parse formal set-op, l-val and r-val
  at_index = expr.find('=')
  if expr[at_index - 1] != ' ':
    if expr[at_index - 1] in ('<', '>', '+', ':', '_', '*', '|', '&'):
      props['reduce_type'] = expr[at_index - 1]
      lval = expr[:at_index - 1].strip()
    else:
      blank_index = expr.find(' ', 0, at_index)
      assert blank_index > 0, "Illegal reduce naming in equation near: `L-value <reduce_type>=`"
      props['reduce_type'] = expr[blank_index + 1:at_index]
      lval = expr[:blank_index].strip()
  else:
    lval = expr[:at_index].strip()
  rval = expr[at_index + 1:].strip()

  # Distinguish data/reduce axes according to l-val
  for x in lval[lval.index('[') + 1:lval.rindex(']')].split(','):
    x = x.strip()
    props['data_axes'].append(scale_ax_name if x == '0' else x)
  for x in explicit_range:
    if x not in props['data_axes']:
      props['reduce_axes'].append(x)

  global full_tensor_dict
  for input_name in full_tensor_dict:
    if not input_name[0].islower():
      raise Exception("Tensor variable name must start with lower case letter: %s" % input_name)
    exec('%s = OpTensor("tensor", input_name, "%s")' % (input_name, full_tensor_dict[input_name]["dtype"]))
    
  # Build ast according to rval & fill uncertain axis range
  _root = eval(rval)
  for x in explicit_range:
    if explicit_range[x] is None:
      raise Exception("The range of axis `%s` is undeterminzed, please use `where` clause to set the range explicitly." % x)

  # Collect output shape inferences
  props['data_axes'] = [{'name': x, 'range': explicit_range[x][1]} for x in props['data_axes']]
  props['reduce_axes'] = [{'name': x, 'range': explicit_range[x][1]} for x in props['reduce_axes']]

  output_name = lval[:lval.index('[')].strip()
  props['output_name'] = output_name

  ast = {'props': props, 'root': _root}

  input_names = set()
  def scan_items(root, ancestor, input_names):
    if root._op == 'tensor':
      input_names.add(root._value)
    elif root._op == 'get_item':
      input_names.add(root._value['tensor']._value)
  walk_in_ast(ast, 'root', scan_items, [input_names,])

  local_input_dict = {}
  for name in input_names:
    local_input_dict[name] = full_tensor_dict[name]
  props['input_dict'] = local_input_dict
  props['explicit_range'] = copy.deepcopy(explicit_range)
  return ast

def const(other, dtype=None):
  return OpTensor.parse(other, output_dtype=dtype)

def alter(other, dtype=None):
  k, v = other.split(':')
  v = v.strip()
  assert re.match(r'^[0-9\.]+$', v)
  return const(eval(v), dtype=dtype).alter(k)

def f_op(func, *args):
  assert len(args) > 0
  return args[0].call(func, list(args[1:]))

def emit_antares_ir(ast, primal=False, tensor_remap=dict()):
  primal_ids = {"axis_id": 0, "tensor_id": 0}
  axis_dict, tensor_dict = {}, {}
  dummy_range = set()

  def _emit(node):
    if node._op == 'const':
      return 'const(%s)' % node._value
    elif node._op == 'axis':
      _value = node._value
      if primal:
        if _value not in axis_dict:
          axis_dict[_value] = '$X%d' % primal_ids['axis_id']
          primal_ids['axis_id'] += 1
        _value = axis_dict[_value]
      if hasattr(node, '_func'):
        return node._func(_value)
      return _value
    elif node._op == 'op':
      if len(node._value['inputs']) == 2:
        return '(%s %s %s)' % (_emit(node._value['inputs'][0]), node._value['name'], _emit(node._value['inputs'][1]))
      raise
    elif node._op == 'get_item':
      _value = node._value['tensor']._value
      _value = tensor_remap.get(_value, _value)
      if primal:
        if _value not in tensor_dict:
          tensor_dict[_value] = '$i%d' % primal_ids['tensor_id']
          primal_ids['tensor_id'] += 1
        _value = tensor_dict[_value]
      for i, ch in enumerate(node._value['index']):
        if ch._op != 'axis':
          continue
        input_size = ast['props']['input_dict'][node._value['tensor']._value]['shape'][i]
        access_size = [x['range'] for x in (ast['props']['data_axes'] + ast['props']['reduce_axes']) if x['name'] == ch._value][0]
        if input_size == access_size:
          dummy_range.add(ch._value)
      return '%s[%s]' % (_value, ', '.join([_emit(x) for x in node._value['index']]))
    elif node._op == 'call':
      if len(node._value['inputs']) == 1:
        return '(%s).call(`%s`, dtype=`%s`)' % (_emit(node._value['inputs'][0]), node._value['name'], node._dtype)
      return '(%s).call(`%s`, [%s], dtype=`%s`)' % (_emit(node._value['inputs'][0]), node._value['name'], ', '.join([_emit(x) for x in node._value['inputs'][1:]]), node._dtype)
    elif node._op == 'when':
      if len(node._value['if']) == 0:
        return '(%s)' % _emit(node._value['true'])
      return '(%s).when([%s], %s, merge_op="%s")' % (_emit(node._value['true']), ', '.join([_emit(x) for x in node._value['if']]), _emit(node._value['false']), node._value['merge_op'])
    elif node._op == 'cast':
      return '(%s).cast(`%s`)' % (_emit(node._value['inputs'][0]), node._dtype)
    elif node._op == 'axis_range':
      return '%s.val()' % (node._value)
    else:
      raise Exception("Emit Antares IR: Unhanled reverse-emit op type: %s" % node._op)

  _value = ast['props']['output_name']
  _value = tensor_remap.get(_value, _value)

  output_name = '$o0' if primal else _value
  lval = '%s[%s]' % (output_name, ', '.join([x['name'] for x in ast['props']['data_axes']]))

  body = _emit(ast['root'])
  comp_type = (ast['props']['reduce_type'] if ast['props']['reduce_type'] else '') + '='

  explicit_axes = [x for x in ast['props']['data_axes'] + ast['props']['reduce_axes'] if x['name'] not in dummy_range]
  if len(explicit_axes) > 0:
    return '%s %s %s where %s' % (lval, comp_type, body, ', '.join(['%s in %d' % (x['name'], x['range']) for x in explicit_axes]))
  else:
    return '%s %s %s' % (lval, comp_type, body)

def walk_in_ast(parent, attr_id, func, args):
  node = getattr(parent, attr_id) if isinstance(parent, OpTensor) else parent[attr_id]

  def _walk(node, parent, attr_id):
    updated_node = func(node, (parent, attr_id), *args)
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
    elif node._op in ['axis', 'const', 'axis_range', 'alter', 'tensor']:
      pass
    else:
      raise Exception('Unhandled node type in walk_in_ast(): %s' % node._op)

  _walk(node, parent, attr_id)

def ir_graph_parser(exprss, input_dict, extra_outputs=[]):
  statements = [s_.strip() for s_ in exprss.split(';')]

  global full_tensor_dict, explicit_range, vamap_tensor

  vamap_tensor = {}
  for key in input_dict:
    input_shape = input_dict[key]['shape']
    vamap_tensor[key] = [None] * len(input_shape)
    for i in range(len(input_shape)):
      if isinstance(input_shape[i], str):
        arg, val = input_shape[i].split(':')
        assert re.match(r'[A-Za-z]+', arg), 'Invalid arg name setting: "%s"' % arg
        vamap_tensor[key][i] = arg
        input_shape[i] = int(val)
      else:
        vamap_tensor[key][i] = int(input_shape[i])

  visited_inputs = set()
  full_tensor_dict = copy.deepcopy(input_dict)
  output_dict = {}
  ast_seq = []
  for s in statements:
    if not s:
      continue
    explicit_range = None
    ast = parse_to_ast(s)
    ast['props']['raw_exprss'] = s

    for x in ast['props']['input_dict']:
      if x in input_dict:
        visited_inputs.add(x)

    k = ast['props']['output_name']
    full_tensor_dict[k] = {"shape": [x['range'] for x in ast['props']['data_axes']], "dtype": ast['root']._dtype}

    vamap_tensor[k] = [explicit_range[x['name']][0] for x in ast['props']['data_axes']]

    if k in extra_outputs:
      output_dict[k] = full_tensor_dict[k]
    ast_seq.append(ast)

  # os.environ['VAMAP_TENSOR'] = json.dumps(vamap_tensor)
  # os.environ['TENSORS_POOL'] = json.dumps(full_tensor_dict)

  # Also include the last output
  if k not in extra_outputs:
    output_dict[k] = full_tensor_dict[k]

  input_dict = dict([(x, input_dict[x]) for x in visited_inputs])

  # import importlib
  # passes = [(x[:-3], 'lang.pass') for x in os.listdir(os.path.dirname(__file__) + '/pass') if x.endswith('.py')]
  # backend_pass_dir = 'backends/%s/pass' % os.environ['BACKEND']
  # if os.path.isdir(backend_pass_dir):
  #   passes += [(x[:-3], backend_pass_dir.replace('/', '.')) for x in os.listdir(backend_pass_dir) if x.endswith('.py')]
  # passes.sort()
  # for pas in passes:
  #   pass_stage = importlib.import_module('%s.%s' % (pas[1], pas[0]))
  #   pass_stage.run_pass_v2(ast_seq, input_dict, output_dict)
  return ast_seq, input_dict, output_dict, full_tensor_dict


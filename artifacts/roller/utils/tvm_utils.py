import tvm
import tvm.tir as tir

def build_name2val(op, shape):
    axis = op.axis
    reduce_axis = op.reduce_axis

    ret = {}
    for i in range(len(axis)):
        ret[axis[i].var.name] = shape[i]
    for i in range(len(reduce_axis)):
        ret[reduce_axis[i].var.name] = shape[i + len(axis)]

    return ret

def extract_producer_load(expr):
    if isinstance(expr, tvm.tir.ProducerLoad):
        return [expr]
    elif isinstance(expr, tvm.tir.Mul):
        return extract_producer_load(expr.a) + extract_producer_load(expr.b)
    elif isinstance(expr, tvm.tir.Add):
        return extract_producer_load(expr.a) + extract_producer_load(expr.b)
    elif isinstance(expr, tvm.tir.Sub):
        return extract_producer_load(expr.a) + extract_producer_load(expr.b)
    elif isinstance(expr, tvm.tir.LT):
        return extract_producer_load(expr.a) + extract_producer_load(expr.b)
    elif isinstance(expr, tvm.tir.expr.Div):
        return extract_producer_load(expr.a) + extract_producer_load(expr.b)
    elif isinstance(expr, tvm.tir.expr.And):
        return extract_producer_load(expr.a) + extract_producer_load(expr.b)
    elif isinstance(expr, tvm.tir.expr.LE):
        return extract_producer_load(expr.a) + extract_producer_load(expr.b)
    elif isinstance(expr, tvm.tir.expr.NE):
        return extract_producer_load(expr.a) + extract_producer_load(expr.b)
    elif isinstance(expr, tvm.tir.expr.EQ):
        return extract_producer_load(expr.a) + extract_producer_load(expr.b)
    elif isinstance(expr, tvm.tir.expr.FloorMod):
        return extract_producer_load(expr.a) + extract_producer_load(expr.b)
    elif isinstance(expr, tvm.tir.Call):
        ret = []
        for arg in expr.args:
            # print('[debug] arg = {}'.format(arg))
            ret = ret + extract_producer_load(arg)
        return ret
    elif isinstance(expr, tvm.tir.FloatImm):
        return []
    elif isinstance(expr, tvm.tir.IntImm):
        return []
    elif isinstance(expr, tvm.tir.StringImm):
        return []
    elif isinstance(expr, tvm.tir.expr.Var):
        return []
    elif isinstance(expr, tvm.tir.expr.Cast):
        return extract_producer_load(expr.value)
    else:
        print('type: {}, content: {}'.format(type(expr), expr))
        assert(False)

def eval_index_len(index_expr, name2val):
    if isinstance(index_expr, tvm.tir.Var):
        return name2val[index_expr.name] - 1
    elif isinstance(index_expr, tvm.tir.Add):
        return eval_index_len(index_expr.a, name2val) + eval_index_len(index_expr.b, name2val)
    elif isinstance(index_expr, tvm.tir.Sub):
        assert(isinstance(index_expr.b, tvm.tir.IntImm))
        return eval_index_len(index_expr.a, name2val)
    elif isinstance(index_expr, tvm.tir.expr.Mul):
        return eval_index_len(index_expr.a, name2val) * eval_index_len(index_expr.b, name2val)
    elif isinstance(index_expr, tvm.tir.expr.Div):
        return eval_index_len(index_expr.a, name2val) // eval_index_len(index_expr.b, name2val)
    elif isinstance(index_expr, tvm.tir.expr.FloorMod):
        return eval_index_len(index_expr.a, name2val) % eval_index_len(index_expr.b, name2val)
    elif isinstance(index_expr, tvm.tir.expr.IntImm):
        return index_expr.value
    else:
        print('type: {}, content: {}'.format(type(index_expr), index_expr))
        assert(False)

def build_tensors(op, shape):
    name2val = build_name2val(op, shape)

    assert(len(op.body) == 1)
    compute_expr = op.body[0]

    def calc_tensor_dim(load_expr):
        indices = load_expr.indices
        dims = [0] * len(indices)
        for i in range(len(indices)):
            if isinstance(indices[i], tvm.tir.expr.IntImm):
                # corner case: slice
                dims[i] = 1
            else:
                dims[i] = eval_index_len(indices[i], name2val) + 1
        return dims

    def process_load_exprs(load_exprs):
        in_tensors = {}
        for load_expr in load_exprs:
            producer_name = load_expr.producer.name
            dims = calc_tensor_dim(load_expr)
            if producer_name in in_tensors:
                assert(dims == in_tensors[producer_name])
            else:
                in_tensors[producer_name] = dims
        return list(in_tensors.items())

    if isinstance(compute_expr, tvm.tir.Reduce):
        in_tensors = {}
        for in_expr in compute_expr.source:
            load_exprs = extract_producer_load(in_expr)
            for load_expr in load_exprs:
                producer_name = load_expr.producer.name
                dims = calc_tensor_dim(load_expr)
                assert(producer_name not in in_tensors)
                in_tensors[producer_name] = dims

        in_tensors = list(in_tensors.items())
    elif isinstance(compute_expr, tvm.tir.Call):
        # TODO: trick here
        if compute_expr.op.name == 'tir.if_then_else':
            constraints = {}

            cur_name2val = {}
            for item in op.axis:
                cur_name2val[item.var.name] = item.dom.extent.value
            for item in op.reduce_axis:
                cur_name2val[item.var.name] = item.dom.extent.value

            def extract_constraints(cur_expr):
                if isinstance(cur_expr, tvm.tir.expr.And):
                    return extract_constraints(cur_expr.a) + extract_constraints(cur_expr.b)
                elif isinstance(cur_expr, tvm.tir.expr.LT):
                    if isinstance(cur_expr.a, tvm.tir.Var):
                        assert(isinstance(cur_expr.b, tvm.tir.expr.IntImm))
                        return [(cur_expr.a.name, 'high', cur_expr.b.value)]
                    else:
                        return []
                elif isinstance(cur_expr, tvm.tir.expr.GE):
                    if isinstance(cur_expr.a, tvm.tir.Var):
                        assert(isinstance(cur_expr.b, tvm.tir.expr.IntImm))
                        return [(cur_expr.a.name, 'low', cur_expr.b.value)]
                    else:
                        return []
                elif isinstance(cur_expr, tvm.tir.expr.LE):
                    if isinstance(cur_expr.b, tvm.tir.Var):
                        assert(isinstance(cur_expr.a, tvm.tir.expr.IntImm))
                        return [(cur_expr.b.name, 'low', cur_expr.a.value)]
                    else:
                        return []
                else:
                    # print('[debug] extract constraints: {}'.format(cur_expr))
                    return []

            info = extract_constraints(compute_expr.args[0])

            for record in info:
                name, kind, val = record

                assert(name in cur_name2val)

                if kind == 'low':
                    if name in constraints:
                        prev_low, prev_high = constraints[name]
                        constraints[name] = (max(val, prev_low), prev_high)
                    else:
                        constraints[name] = (val, cur_name2val[name])
                else:
                    assert(kind == 'high')
                    if name in constraints:
                        prev_low, prev_high = constraints[name]
                        constraints[name] = (prev_low, min(val, prev_high))
                    else:
                        constraints[name] = (0, val)

            for key, val in constraints.items():
                low, high = val
                assert(low < high)
                assert(key in name2val)
                prev_val = name2val[key]
                # print(prev_val, cur_name2val, low, high)
                name2val[key] = prev_val - (cur_name2val[key] - (high - low))

        load_exprs = []
        for arg in compute_expr.args:
            load_exprs = load_exprs + extract_producer_load(arg)
        in_tensors = {}
        for load_expr in load_exprs:
            producer_name = load_expr.producer.name
            dims = calc_tensor_dim(load_expr)
            if producer_name in in_tensors:
                assert(dims == in_tensors[producer_name])
            else:
                in_tensors[producer_name] = dims
        in_tensors = list(in_tensors.items())
    elif isinstance(compute_expr, tvm.tir.expr.ProducerLoad):
        is_gather = False
        for index in compute_expr.indices:
            if isinstance(index, tvm.tir.expr.ProducerLoad):
                is_gather = True
                break

        if is_gather:
            in_tensors = [(compute_expr.producer.name, shape)]
            for index in compute_expr.indices:
                if isinstance(index, tvm.tir.expr.ProducerLoad):
                    in_tensors.append((index.producer.name, calc_tensor_dim(index)))
        else:
            in_tensors = [(compute_expr.producer.name, calc_tensor_dim(compute_expr))]
    elif isinstance(compute_expr, tvm.tir.expr.Add) or isinstance(compute_expr, tvm.tir.expr.Div):
        load_exprs = extract_producer_load(compute_expr.a) + extract_producer_load(compute_expr.b)
        in_tensors = {}
        for load_expr in load_exprs:
            producer_name = load_expr.producer.name
            dims = calc_tensor_dim(load_expr)
            if producer_name in in_tensors:
                assert(dims == in_tensors[producer_name])
            else:
                in_tensors[producer_name] = dims
        in_tensors = list(in_tensors.items())
    elif isinstance(compute_expr, tvm.tir.expr.Cast):
        load_exprs = extract_producer_load(compute_expr.value)
        in_tensors = process_load_exprs(load_exprs)
    else:
        print('WARNING: type: {}, content: {}'.format(type(compute_expr), compute_expr))
        assert(False)

    out_tensors = [(op.output(0).name, shape[:len(op.axis)])]

    return in_tensors, out_tensors

def extract_iter_names(index_expr):
    if isinstance(index_expr, tvm.tir.Var):
        return [index_expr.name]
    elif isinstance(index_expr, tvm.tir.Add):
        return extract_iter_names(index_expr.a) + extract_iter_names(index_expr.b)
    elif isinstance(index_expr, tvm.tir.expr.Mul):
        return extract_iter_names(index_expr.a) + extract_iter_names(index_expr.b)
    elif isinstance(index_expr, tvm.tir.expr.Div):
        return extract_iter_names(index_expr.a) + extract_iter_names(index_expr.b)
    elif isinstance(index_expr, tvm.tir.expr.FloorMod):
        return extract_iter_names(index_expr.a) + extract_iter_names(index_expr.b)
    elif isinstance(index_expr, tvm.tir.expr.IntImm):
        return []
    else:
        print(type(index_expr))
        assert(False)

def get_normalized_reduce_axis(op):
    if len(op.reduce_axis) == 0:
        return op.reduce_axis
    lhs, rhs = [], []

    assert(len(op.body) == 1)
    compute_expr = op.body[0]

    not_naive_access = set()
    def process_load_exprs(load_exprs):
        for expr in load_exprs:
            for dim_expr in expr.indices:
                if not isinstance(dim_expr, tvm.tir.Var):
                    for name in extract_iter_names(dim_expr):
                        not_naive_access.add(name)

    if isinstance(compute_expr, tvm.tir.Reduce):
        for in_expr in compute_expr.source:
            load_exprs = extract_producer_load(in_expr)
            process_load_exprs(load_exprs)
    elif isinstance(compute_expr, tvm.tir.Call):
        load_exprs = []
        for arg in compute_expr.args:
            load_exprs = load_exprs + extract_producer_load(arg)
        process_load_exprs(load_exprs)
    elif isinstance(compute_expr, tvm.tir.expr.ProducerLoad):
        process_load_exprs([compute_expr])
    else:
        print('type: {}, content: {}'.format(type(compute_expr), compute_expr))
        assert(False)

    # print('[debug] not naive access: {}'.format(not_naive_access))

    for axis in op.reduce_axis:
        if axis.var.name in not_naive_access:
            rhs.append(axis)
        else:
            lhs.append(axis)

    return lhs + rhs


def classify_tvm_op_tensors(op):
    in_tensors, out_tensors = [], []

    out_name = op.output(0).name

    def dfs(tvm_op):
        if isinstance(tvm_op, tvm.te.tensor.PlaceholderOp):
            in_tensors.append(tvm_op.output(0))
        elif isinstance(tvm_op, tvm.te.tensor.ComputeOp):
            for tensor in tvm_op.input_tensors:
                dfs(tensor.op)

            for idx in range(tvm_op.num_outputs):
                tensor = tvm_op.output(idx)
                if tensor.name == out_name or tensor.name + '_unpad' == out_name:
                    out_tensors.append(tensor)
                else:
                    in_tensors.append(tensor)
        else:
            print('type: {}, content: {}'.format(type(tvm_op), tvm_op))
            assert(False)

    dfs(op)
    return in_tensors, out_tensors

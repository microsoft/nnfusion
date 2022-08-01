import tvm

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
    elif isinstance(expr, tvm.tir.LT):
        return extract_producer_load(expr.a) + extract_producer_load(expr.b)
    elif isinstance(expr, tvm.tir.expr.Div):
        return extract_producer_load(expr.a) + extract_producer_load(expr.b)
    elif isinstance(expr, tvm.tir.FloatImm):
        return []
    else:
        print(type(expr))
        assert(False)

def eval_index_len(index_expr, name2val):
    if isinstance(index_expr, tvm.tir.Var):
        return name2val[index_expr.name]
    elif isinstance(index_expr, tvm.tir.Add):
        return eval_index_len(index_expr.a, name2val) + eval_index_len(index_expr.b, name2val) - 1
    elif isinstance(index_expr, tvm.tir.expr.Mul):
        return (eval_index_len(index_expr.a, name2val) - 1) * (eval_index_len(index_expr.b, name2val) - 1) + 1
    elif isinstance(index_expr, tvm.tir.expr.IntImm):
        return index_expr.value + 1
    else:
        print(type(index_expr))
        assert(False)

def build_tensors(op, shape):
    name2val = build_name2val(op, shape)

    assert(len(op.body) == 1)
    compute_expr = op.body[0]

    def calc_tensor_dim(load_expr):
        indices = load_expr.indices
        dims = [0] * len(indices)
        for i in range(len(indices)):
            dims[i] = eval_index_len(indices[i], name2val)
        return dims

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
    else:
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
    elif isinstance(index_expr, tvm.tir.expr.IntImm):
        return []
    else:
        print(type(index_expr))
        assert(False)

def get_normalized_reduce_axis(op):
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
    else:
        assert(False)

    # print('[debug] not naive access: {}'.format(not_naive_access))

    for axis in op.reduce_axis:
        if axis.var.name in not_naive_access:
            rhs.append(axis)
        else:
            lhs.append(axis)

    return lhs + rhs

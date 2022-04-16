import tvm
from threading import Thread

def get_axis_names(Tensor):
    s = tvm.te.create_schedule(Tensor.op)
    saxis = [axis.var.name for axis in s[Tensor].op.axis]
    raxis = [axis.var.name for axis in s[Tensor].op.reduce_axis]
    return saxis, raxis

def str_to_ms(string):
    if string.endswith("ms"):
        return float(string[:-2])
    elif string.endswith("us"):
        return float(string[:-2]) / 1000
    elif string.endswith("s"):
        return float(string[:-1]) * 1000

def get_time_from_nvprof_file(out, backend="tvm"):    
    with open(out, "r") as inf:
        lines = inf.readlines()
        if backend == "tvm":
            kernel_name = "default_function_kernel0"
        if backend == "antares":
            kernel_name = "template_op_kernel0"
        for line in lines:
            if kernel_name in line:
                breaks = line.split()
                return str_to_ms(breaks[-4])
    # kernel does not execute
    return 1e100

def get_time_from_rocm_file(file_name="_tmp"):
    try:
        with open(file_name) as f:
            for line in f.readlines():
                if "- TPR" in line:
                    t_ms = float(line.rstrip()[7:])
                    return t_ms
        return 1e100
    except:
        return 1e100

def simplify(expr):
    """Simplify the expression if it is Expr, directly return if it is int.
    Parameters
    ----------
    expr : Expr or int
        The input.
    Returns
    -------
    out : Expr or int
        The simplified output
    """
    return tvm.arith.Analyzer().simplify(expr) if isinstance(expr, tvm.tir.PrimExpr) else expr


def get_pad_tuple(padding, kernel):
    """Common code to get the pad option
    Parameters
    ----------
    padding : int or str
        Padding size, or ['VALID', 'SAME']
    kernel : tuple of int
        Conv kernel size
    Returns
    -------
    pad_top : int
        Padding size on top
    pad_left : int
        Padding size on left
    pad_down : int
        Padding size on down.
    pad_right : int
        Padding size on right.
    """
    # compute the padding size
    if isinstance(padding, (tuple, list)):
        if len(padding) == 2:
            pad_h = padding[0] * 2
            pad_w = padding[1] * 2
        elif len(padding) == 4:
            return padding[0], padding[1], padding[2], padding[3]
        else:
            raise ValueError("Size of padding can only be 2 or 4")
    elif isinstance(padding, int):
        pad_h = pad_w = padding * 2
    elif padding == "VALID":
        pad_h = 0
        pad_w = 0
    elif padding == "SAME":
        pad_h = kernel[0] - 1
        pad_w = kernel[1] - 1
    else:
        raise ValueError("Unknown padding option %s" % padding)
    pad_top = (pad_h + 1) // 2
    pad_left = (pad_w + 1) // 2
    return pad_top, pad_left, pad_h - pad_top, pad_w - pad_left


def alloc_configs_for_subprocess(parallel, configs_num):
    num_process = [int(configs_num // parallel) + 1 for i in range(configs_num % parallel)]
    num_process = num_process + [int(configs_num // parallel) for i in range(parallel - configs_num % parallel)]
    idx = 0
    process_idx = [0]
    for num in num_process:
        process_idx.append(idx + num)
        idx += num
    return process_idx


class MyThread(Thread):
    def __init__(self, func, args):
        super(MyThread, self).__init__()
        self.func = func
        self.args = args

    def run(self):
        self.result = self.func(*self.args)

    def get_result(self):
        try:
            return self.result
        except Exception:
            return None

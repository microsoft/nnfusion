"""
The entry of Antares-TVM bridge
"""
import threading
from collections import OrderedDict
from typing import List, Optional, Tuple

import numpy as np
from tvm import te, tir

from . import einstein_v2 as einstein_core

OUTPUT_TEMP = list()
INPUT_TEMP = list()
_lock = threading.Lock()

def einstein_v2(exprss: str, input_dict: List, extra_outputs: Optional[List] = [], **kwargs):
    """
    The function to translate Antares to TVM IR

    Args:
        exprss (str): the Antares IR expression
        input_dict (List): input dict
    """
    del kwargs
    if isinstance(input_dict, (list, )):
        ordered = OrderedDict()
        for i in input_dict:
            ordered[i[0]] = i[1]
        input_dict = ordered
    for k in input_dict:
        if len(input_dict[k]['shape']) == 0:
            input_dict[k]['shape'] = [1]
    antares_ir = einstein_core.emit_tvm_ir(exprss, input_dict, extra_outputs)
    assert len(antares_ir) > 0
    exec(antares_ir, globals())  # pylint:disable=exec-used


def common_reduce(name, args=(0, )):
    """
    The reduce op parse from Antares IR to TVM
    """
    if not isinstance(args, tuple) and not isinstance(args, list):
        args = (args, )

    def _reduce_op(_x, _y):
        assert _x.dtype == _y.dtype
        return tir.call_pure_extern(_x.dtype, name, _x, _y, *args[1:])

    return te.comm_reducer(_reduce_op,
                           lambda t: tir.const(args[0], dtype=t),
                           name=name)


def cast_dtype(dtype):
    """
    The dtype parse from Antares IR to TVM
    """
    idx = dtype.find('@')
    if idx < 0:
        return dtype
    else:
        print(dtype)
        raise NotImplementedError


def input(name, shape, dtype="float32"):  # pylint:disable=redefined-builtin
    """
    The input parse from Antares IR to TVM
    """
    global INPUT_TEMP  # pylint:disable=global-statement
    if len(shape) == 0:
        shape = [1]
    result = te.placeholder(shape, dtype=dtype, name=name)
    INPUT_TEMP.append(result)
    return result


def loop(length, name="rv", start=0):
    """
    Reduce axis definition
    """
    return te.reduce_axis((start, length), name=name)


def output(shape,  # pylint:disable=too-many-arguments
           func=None,
           flops=None,
           name='output0',
           topi=None,
           dtype=None,
           tag='',
           final_output=True):
    """
    The output parse from Antares IR to TVM
    """
    global OUTPUT_TEMP  # pylint:disable=global-statement
    if len(shape) == 0:
        shape = [1]
    if flops is None:
        flops = np.product(shape)
    if topi is not None:
        result = te.compute(topi.shape, lambda *X: topi[X], name=name, tag='')
    else:
        # print(name, shape, func, dtype, tag)
        result = te.compute(shape, func, name=name, tag=tag)

    if not shape:
        shape = result.shape
    if not dtype:
        dtype = result.dtype

    if final_output:
        OUTPUT_TEMP.append(result)
    return result

def translate_to_tvm(expr: str, input_dict, extra_outputs=[]) -> Tuple[List[te.Tensor], List[te.Tensor]]:
    _lock.acquire()
    OUTPUT_TEMP.clear()
    INPUT_TEMP.clear()
    einstein_v2(expr, input_dict, extra_outputs)
    input_args, output_args = INPUT_TEMP.copy(), OUTPUT_TEMP.copy()
    _lock.release()
    return input_args, output_args

def translate_ir_to_tvm(antares_ir: str) -> Tuple[List[te.Tensor], List[te.Tensor]]:
    antares_ir = antares_ir.strip()
    assert antares_ir.startswith(
        '- '
    ), "The computing expression doesn't start with proper prefix: - ..."

    antares_ir = antares_ir[2:]
    antares_ir = antares_ir.replace("einstein_v2", "translate_to_tvm")
    input_args, output_args = eval(antares_ir, globals(), locals())
    return input_args, output_args

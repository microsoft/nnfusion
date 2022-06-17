"""
The entry of Antares-TVM bridge
"""
import importlib
from collections import OrderedDict
from typing import Optional, List

import numpy as np
from tvm import te, tir

from . import einstein_v2 as einstein_core

OUTPUT_TEMP = list()
INPUT_TEMP = list()

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
        result = te.compute(shape, func, name=name, tag=tag)

    if not shape:
        shape = result.shape
    if not dtype:
        dtype = result.dtype

    if final_output:
        OUTPUT_TEMP.append(result)
    return result


def do_native_scheduling(sch: te.Schedule, backend: str, plan='default_v2', plan_args=None):
    """
    Scheduler entry
    """
    def _select_plan(plan_name):
        if plan_name.find('.') < 0:
            plan_name = 'standard.' + plan_name
        schedule_lib = importlib.import_module(
            'backends.%s.schedule.%s' % (backend, plan_name), __loader__.name)
        sch_args = plan_args if plan_args is not None else list()
        return schedule_lib.schedule(sch, *sch_args)

    if plan is None:
        raise Exception(f'No available plan configured for backend: {backend}')
    return _select_plan(plan)


def _assign_multiple_outputs(outputs: List[te.Tensor]):
    outputs_shape = [np.asarray(list(o_.shape), dtype=np.int32) for o_ in outputs]
    shared_shape = outputs_shape[np.argsort([s_.size for s_ in outputs_shape])[0]]

    def _alignment(dst_shape, src_shape, src_args):
        dst_shape = np.asarray(list(dst_shape), dtype=np.int32)
        dst_args = list()
        src_index = 0
        for dst_ in dst_shape:
            if dst_ == 1:
                dst_args.append(0)
                continue
            dst_args.append(src_args[src_index])
            src_index += 1
        return tuple(dst_args)

    def _shared_assignment(*args):
        assert len(args) == shared_shape.size
        out_selects = list()
        for out_tensor in outputs:
            out_args = _alignment(out_tensor.shape, shared_shape, args)
            out_selects.append(out_tensor[out_args])
        return out_selects

    assigned_outputs = te.compute(tuple(shared_shape.tolist()), _shared_assignment,
                                  name='proxy_outputs')
    return assigned_outputs

def translate_to_tvm(expr, input_dict, extra_outputs=[]):
    OUTPUT_TEMP.clear()
    INPUT_TEMP.clear()
    einstein_v2(expr, input_dict, extra_outputs)
    return INPUT_TEMP, OUTPUT_TEMP

def translate_ir_to_tvm(antares_ir):
    antares_ir = antares_ir.strip()
    assert antares_ir.startswith(
        '- '
    ), "The computing expression doesn't start with proper prefix: - ..."

    antares_ir = antares_ir[2:]
    antares_ir = antares_ir.replace("einstein_v2", "translate_to_tvm")
    input_args, output_args = eval(antares_ir, globals(), locals())
    return input_args.copy(), output_args.copy()

import torch

from ast_analyzer.python_std.run_py_passes import run_model, apply_passes_recursive

from ast_analyzer.shape_inference.type_inference import *
from ast_analyzer.shape_inference.type_inference_tools import (
    print_inference_results, type_inference_model)

from .utils.timer import Timer
from time import time
from ast_analyzer.tensor_opt.remove_ret import remove_ret
from ast_analyzer.tensor_opt import buttom_up_feed

from collections import Iterable

import importlib
from ast_analyzer.utils.nvprof import profile_start, profile_stop, enable_profile
from ast_analyzer.tensor_opt.search_best_flags import search_best_flags

def check_equal(ref, out, allow_233=False):
    precision = 1e-3
    if isinstance(ref, torch.Tensor):
        assert(isinstance(out, torch.Tensor))
        r = ref.cpu()
        o = out.cpu()
        if r.dtype == torch.bool and o.dtype == torch.int8:
            o = o.bool()
        all_close = torch.allclose(r, o, atol=precision, rtol=precision)
        if all_close:
            print("tensor equals!")
        elif allow_233 and torch.max(out) == 233.0 and torch.min(out) == 233.0:
            print("result is 233")
        else:
            close = torch.isclose(r, o, rtol=precision, atol=precision)
            print("ref:", torch.masked_select(r, ~close))
            print("out:", torch.masked_select(o, ~close))
            print(torch.sum(~close))
            print("wrong answer !!!!!!!!!!!!!!!!!!!!!!!!!!")
            assert(False)
    elif isinstance(ref, Iterable):
        assert(isinstance(out, Iterable))
        for r, o in zip(ref, out):
            check_equal(r, o, allow_233)


def test_torch_eval(model, inp, profile=None):
    print("[pytorch]")
    n_warmup = 100
    n_run = 100
    print("[warmup]")
    torch.cuda.synchronize()
    for i in range(n_warmup):
        t0 = time()
        output = model.forward(*inp)
        torch.cuda.synchronize()
        print("Time {} ms".format((time() - t0) * 1000))

    timer = Timer("ms")
    torch.cuda.synchronize()
    if profile == "pytorch":
        profile_start()
    for i in range(n_run):
        timer.start()
        output = model.forward(*inp)
        torch.cuda.synchronize()
        timer.log()
    if profile == "pytorch":
        profile_stop()
    timer.report()

    print("[TorchScript]")
    m = torch.jit.script(model)
    print("[warmup]")
    torch.cuda.synchronize()
    for i in range(n_warmup):
        t0 = time()
        output = m.forward(*inp)
        torch.cuda.synchronize()
        print("Time {} ms".format((time() - t0) * 1000))

    timer = Timer("ms")
    torch.cuda.synchronize()
    if profile == "torchscript":
        profile_start()
    for i in range(n_run):
        timer.start()
        output = m.forward(*inp)
        torch.cuda.synchronize()
        timer.log()
    if profile == "torchscript":
        profile_stop()
    timer.report()

    # torch.onnx.export(m, inp, "tmp/model.onnx", verbose=True, opset_version=11,)

def get_ref(model, inp):
    ref = model.forward(*inp)
    if isinstance(ref, tuple):
        ref = tuple([r.clone() for r in ref])
    torch.cuda.empty_cache()
    return ref


def get_modules(model, model_name, inp, platform, run_unroll, enable_control_flow):
    buttom_up_feed.ENABLE_CONTROL_FLOW = enable_control_flow
    if not enable_control_flow:
        buttom_up_feed.SEARCH_ALL_SUBAST = True
    if run_unroll:
        node = run_model(model)
    else:
        node = utils.get_ast(model.forward)
        assert(len(node.body) == 1)
        node.body[0].name = 'forward'
        model.__ast__ = node

    node2type = type_inference_model(model, inp) # build the call graph

    apply_passes_recursive(node, node2type, set([model.forward]))

    node2type = type_inference_model(model, inp)
    _, _, node, _, to_import_list = buttom_up_feed.buttom_up_feed(node, model_name, node2type, model, model.forward, platform)

    to_import_list = [x for x in to_import_list if x is not None]
    return to_import_list


def measure(model, inp, n_warmup, n_run, platform):
    enable_profile(platform)
    print("[warmup]")
    torch.cuda.synchronize()
    for i in range(n_warmup):
        t0 = time()
        out = model.forward(*inp)
        torch.cuda.synchronize()
        print("Time {} ms".format((time() - t0) * 1000))
    timer = Timer("ms")
    torch.cuda.synchronize()
    profile_start(platform)
    for i in range(n_run):
        timer.start()
        output_nnf = model.forward(*inp)
        torch.cuda.synchronize()
        timer.log()
    profile_stop(platform)
    timer.report()


def workflow_fix_flag(model, model_name, inp, platform, time_measure=False, allow_233=False, free=False, run_unroll=False, enable_control_flow=True):
    ref = get_ref(model, inp)
    to_import_list = get_modules(model, model_name, inp, platform, run_unroll, enable_control_flow)

    imported = []

    for scope, func_name, model_name, onnx_model in to_import_list:
        module_dir = f'{model_name}.Model_nnf_fix_flag'
        torch_func = importlib.import_module(module_dir)
        imported.append(torch_func)
        setattr(scope, func_name, torch_func.GenModel)
    inp_tensors = []
    for x in inp:
        if isinstance(x, torch.Tensor):
            inp_tensors.append(x)
        else:
            inp_tensors.append(torch.full((), x, device='cuda'))
    inp = tuple(inp_tensors)
    out = model.forward(*inp)

    check_equal(ref, out, allow_233)

    n_warmup = n_run = 100

    if time_measure:
        measure(model, inp, n_warmup, n_run, platform)
    if free:
        for m in imported:
            m.GenModel.freeall()


def workflow_search_flag(model, model_name, inp, platform, time_measure=False, allow_233=False, free=False, run_unroll=False, enable_control_flow=True):
    ref = get_ref(model, inp)
    from ast_analyzer.to_onnx import to_torch_func
    to_torch_func.NNFUSION_CODEGEN_FLAGS = {'TOFILL':'TOFILL'}
    to_import_list = get_modules(model, model_name, inp, platform, run_unroll, enable_control_flow)
    inp_tensors = []
    for x in inp:
        if isinstance(x, torch.Tensor):
            inp_tensors.append(x)
        else:
            inp_tensors.append(torch.full((), x, device='cuda'))
    inp = tuple(inp_tensors)
    search_best_flags(model, inp, to_import_list, platform)

    out = model.forward(*inp)
    check_equal(ref, out, allow_233)

    n_warmup = n_run = 100

    if time_measure:
        measure(model, inp, n_warmup, n_run, platform) 

from ast_analyzer.shape_inference.types import TyTuple
from numpy.lib.arraysetops import isin
import torch

from ast_analyzer.grad.run import grad_func, grad_model
from ast_analyzer.grad import annotations as anno
from ast_analyzer.grad.annotate import mark_shape, resolve_calls, resolve_anno

from ast_analyzer.python_std.inject import inject, inject_fwdbwd
from ast_analyzer.python_std.run_py_passes import (
    zero_fold, advance_dce, apply_passes_ast, fill_shape_func, run, run_function, run_model, apply_passes_recursive)

from ast_analyzer.shape_inference.utils import clip_head
from ast_analyzer.shape_inference.type_inference import *
from ast_analyzer.shape_inference.type_inference_tools import (
    print_inference_results, type_inference_model, type_inference_func, type_inference_fwdbwd, type_inference_fwdbwd_function)
from ast_analyzer.to_onnx.exporter import export_to_onnx_train, export_to_onnx_eval
from ast_analyzer.to_onnx.inject import inject_training
from ast_analyzer.to_onnx.training_to_torch import to_torch_code
import gast
import astunparse
import inspect

import onnx
import onnxruntime as ort

from .utils.timer import Timer
from time import time
from ast_analyzer.tensor_opt.remove_ret import remove_ret
from ast_analyzer.tensor_opt import buttom_up_feed

from ast_analyzer.grad.anf import anf, invariant
from ast_analyzer.grad.simplify import SplitToIndex

import ctypes
from collections import Iterable

import importlib
from ast_analyzer.utils.nvprof import profile_start, profile_stop, enable_profile
from ast_analyzer.tensor_opt.search_best_flags import search_best_flags

from ast_analyzer.utils import config


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


def workflow_eval(loss_func, model, inp, model_name, profile=False, use_nnfusion=True):
    output = model(*inp)
    print(output)
    run_model(model)
    node2type = type_inference_model(model, inp)
    # print_inference_results(node2type)
    export_to_onnx_eval(model, node2type, model_name)
    output_ort = model.forward_onnx(*inp)
    print("[output-ORT]", output_ort)
    check_equal(output, output_ort)

    if not use_nnfusion:
        return

    # output_tf = model.forward_tf(*inp)
    # print("[output-TF]", output_tf)

    # Time measurement
    n_warmup = 100
    n_run = 100
    print("[warmup]")
    torch.cuda.synchronize()
    for i in range(n_warmup):
        t0 = time()
        output_nnf = model.forward_nnf(*inp)
        torch.cuda.synchronize()
        print("Time {} ms".format((time() - t0) * 1000))
        # if i == 0:
        #     print("result:", output_nnf)
        check_equal(output, output_nnf)

    timer = Timer("ms")
    torch.cuda.synchronize()
    if profile:
        profile_start()
    for i in range(n_run):
        timer.start()
        output_nnf = model.forward_nnf(*inp)
        torch.cuda.synchronize()
        timer.log()
    if profile:
        profile_stop()
    timer.report()


def workflow_train(loss_func, model, inp, model_name, device, profile=False, test_sct=False, test_step=False, use_nnfusion=True):
    loss = loss_func(model, *inp)
    # print("loss=", loss)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    optimizer.zero_grad()
    loss.backward()

    reference = {}
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name, param.grad)
            reference[name] = param.grad.clone()

    run_function(loss_func)
    run_model(model)
    node2type = type_inference_func(loss_func, (model, *inp))
    fwd_ret_ty = node2type[model.__ast__.body[0]].retty
    if isinstance(fwd_ret_ty, TyTensor):
        fwd_ret = 1
    elif isinstance(fwd_ret_ty, TyTuple):
        fwd_ret = fwd_ret_ty.size()
    else:
        raise NotImplementedError
    # print_inference_results(node2type)

    fwdbwd, attrs_order = grad_model(model, node2type, device)
    node2type_fwd, node2type_bwd = type_inference_fwdbwd(
        fwdbwd, (model, *inp), model)
    # print_inference_results(node2type_fwd)
    # print_inference_results(node2type_bwd)

    fill_shape_func(fwdbwd.body[0], node2type_fwd)
    fill_shape_func(fwdbwd.body[1], node2type_bwd)

    apply_passes_ast(gast.Module(
        body=[fwdbwd.body[1]], type_ignores=[]), model)
    apply_passes_ast(gast.Module(
        body=[fwdbwd.body[0]], type_ignores=[]), model)

    zero_fold(fwdbwd)

    advance_dce(fwdbwd)

    inject_fwdbwd(model, fwdbwd)
    results = model.grad_forward(*inp)
    # print("[forward output]")
    # for rr in results:
    #     print("**", rr)

    self_attrs = tuple(getattr(model, anno.getanno(node, 'attr_name'))
                       for node in fwdbwd.body[1].args.args if anno.hasanno(node, 'attr_name'))
    args = (torch.ones([1], dtype=torch.float,
            device=device),) + results[fwd_ret:] + self_attrs
    gd = model.grad_backward(*args)
    print("[backward output]")
    for g in gd:
        print("**", g)

    n_warmup = 100
    n_run = 100

    if test_sct:
        print("[SCT]")
        print("[warmup]")
        torch.cuda.synchronize()
        with torch.no_grad():
            for i in range(n_warmup):
                t0 = time()
                results = model.grad_forward(*inp)
                torch.cuda.synchronize()
                t1 = time()
                optimizer.zero_grad()
                torch.cuda.synchronize()
                t2 = time()
                args = (torch.ones([1], dtype=torch.float,
                                   device=device),) + results[fwd_ret:] + self_attrs
                gd = model.grad_backward(*args)
                torch.cuda.synchronize()
                t3 = time()
                print("Time fwd = {} ms zero = {} ms bwd = {} ms sum = {} ms".format(
                    (t1 - t0) * 1000, (t2 - t1) *
                    1000, (t3 - t2) * 1000, (t3 - t0) * 1000
                ))

            if profile:
                profile_start()
            timer = Timer("ms")
            torch.cuda.synchronize()
            for i in range(n_run):
                timer.start()
                results = model.grad_forward(*inp)
                optimizer.zero_grad()
                args = (torch.ones([1], dtype=torch.float,
                                   device=device),) + results[fwd_ret:] + self_attrs
                model.grad_backward(*args)
                torch.cuda.synchronize()
                timer.log()
            if profile:
                profile_stop()
            timer.report()
            exit(0)

    print("[before to_onnx]")
    print(astunparse.unparse(fwdbwd))

    model.__fwd_ast__ = gast.Module(body=[fwdbwd.body[0]], type_ignores=[])
    model.__bwd_ast__ = gast.Module(body=[fwdbwd.body[1]], type_ignores=[])

    node2type_fwd, node2type_bwd = type_inference_fwdbwd(
        fwdbwd, (model, *inp), model)

    export_to_onnx_train(model, node2type_fwd,
                         node2type_bwd, model_name, fwd_ret, attrs_order, use_nnfusion)

    if not use_nnfusion:
        output_nnf = model.forward_onnx(*inp)
    else:
        output_nnf = model.forward_nnf(*inp)

    # print("[output-ORT]")
    # print(output_nnf)
    optimizer.zero_grad()
    output_nnf.backward()
    # for name, param in model.named_parameters():
    #     if param.requires_grad:
    #         print(name, param.grad)
    for name, param in model.named_parameters():
        if param.requires_grad:
            close = torch.isclose(
                param.grad, reference[name], rtol=1e-3, atol=1e-3)
            if torch.any(~close):
                print(torch.masked_select(param.grad, ~close))
                print(torch.masked_select(reference[name], ~close))
                print(torch.masked_select(reference[name], ~close).shape)
            assert(torch.allclose(
                param.grad, reference[name], rtol=1e-3, atol=1e-3))

    if not use_nnfusion:
        return

    print("[warmup]")
    torch.cuda.synchronize()
    for i in range(n_warmup):
        t0 = time()
        output = model.forward(*inp)
        torch.cuda.synchronize()
        t1 = time()
        optimizer.zero_grad()
        torch.cuda.synchronize()
        t2 = time()
        output.backward()
        torch.cuda.synchronize()
        t3 = time()
        print("Time fwd = {} ms zero = {} ms bwd = {} ms sum = {} ms".format(
            (t1 - t0) * 1000, (t2 - t1) *
            1000, (t3 - t2) * 1000, (t3 - t0) * 1000
        ))

    torch.cuda.synchronize()
    if test_step:
        timer_fwd = Timer("ms")
        timer_zero_grad = Timer("ms")
        timer_bwd = Timer("ms")
        for i in range(n_run):
            torch.cuda.synchronize()
            timer_fwd.start()
            output = model.forward(*inp)
            torch.cuda.synchronize()
            timer_fwd.log()
            timer_zero_grad.start()
            optimizer.zero_grad()
            torch.cuda.synchronize()
            timer_zero_grad.log()
            timer_bwd.start()
            output.backward()
            torch.cuda.synchronize()
            timer_bwd.log()
        print("[test_step]")
        timer_fwd.report()
        timer_zero_grad.report()
        timer_bwd.report()

    timer = Timer("ms")
    torch.cuda.synchronize()
    if profile:
        profile_start()
    for i in range(n_run):
        timer.start()
        output = model.forward(*inp)
        optimizer.zero_grad()
        output.backward()
        torch.cuda.synchronize()
        timer.log()
    if profile:
        profile_stop()
    print("[Training time]")
    timer.report()


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


def test_torch_train(model, inp, profile=None):
    print("[pytorch]")
    n_warmup = 100
    n_run = 100
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    print("[warmup]")
    torch.cuda.synchronize()
    for i in range(n_warmup):
        t0 = time()
        output = model.forward(*inp)
        torch.cuda.synchronize()
        t1 = time()
        optimizer.zero_grad()
        torch.cuda.synchronize()
        t2 = time()
        output.backward()
        torch.cuda.synchronize()
        t3 = time()
        print("Time fwd = {} ms zero = {} ms bwd = {} ms sum = {} ms".format(
            (t1 - t0) * 1000, (t2 - t1) *
            1000, (t3 - t2) * 1000, (t3 - t0) * 1000
        ))

    timer = Timer("ms")
    torch.cuda.synchronize()
    if profile == "pytorch":
        profile_start()
    for i in range(n_run):
        timer.start()
        output = model.forward(*inp)
        optimizer.zero_grad()
        output.backward()
        torch.cuda.synchronize()
        timer.log()
    if profile == "pytorch":
        profile_stop()
    timer.report()

    print("[TorchScript]")
    m = torch.jit.script(model)
    n_warmup = 100
    n_run = 100
    optimizer = torch.optim.SGD(m.parameters(), lr=0.01, momentum=0.9)
    print("[warmup]")
    torch.cuda.synchronize()
    for i in range(n_warmup):
        t0 = time()
        output = m.forward(*inp)
        torch.cuda.synchronize()
        t1 = time()
        optimizer.zero_grad()
        torch.cuda.synchronize()
        t2 = time()
        output.backward()
        torch.cuda.synchronize()
        t3 = time()
        print("Time fwd = {} ms zero = {} ms bwd = {} ms sum = {} ms".format(
            (t1 - t0) * 1000, (t2 - t1) *
            1000, (t3 - t2) * 1000, (t3 - t0) * 1000
        ))

    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    timer = Timer("ms")
    torch.cuda.synchronize()
    if profile == "torchscript":
        profile_start()
    for i in range(n_run):
        timer.start()
        output = m.forward(*inp)
        optimizer.zero_grad()
        output.backward()
        torch.cuda.synchronize()
        timer.log()
    if profile == "torchscript":
        profile_stop()
    timer.report()


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

    # for scope, func_name, model_name, onnx_model in to_import_list:
    #     module_dir = f'{model_name}.Model_onnx'
    #     torch_func = importlib.import_module(module_dir)
    #     imported.append(torch_func)
    #     setattr(scope, func_name, torch_func.GenModel)
    # inp_tensors = []
    # for x in inp:
    #     if isinstance(x, torch.Tensor):
    #         inp_tensors.append(x)
    #     else:
    #         inp_tensors.append(torch.full((), x, device='cuda'))
    # inp = tuple(inp_tensors)
    # # remove_ret(node, model.forward)
    # out = model.forward(*inp)
    # print("out:", out)

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
    # remove_ret(node, model.forward)
    out = model.forward(*inp)
    print("out:", out)
    # print("out.shape", out.shape)
    # print("ref.shape", ref.shape)
    # print(out)

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

def run_reference(model, inp, optimizer):
    loss = model(*inp)
    # print("loss=", loss)
    optimizer.zero_grad()
    loss.backward()

    reference = {}
    for name, param in model.named_parameters():
        if param.requires_grad:
            # print(name, param.grad)
            reference[name] = param.grad.clone()

    return loss, reference


def get_opt_ast_and_type(model, inp):
    node = utils.get_ast(model.forward)
    model.__ast__ = node
    node2type = type_inference_model(model, inp) # build the call graph
    apply_passes_recursive(node, node2type, set([model.forward]))
    node2type = type_inference_model(model, inp)
    # print_inference_results(node2type)

    fwd_ret_ty = node2type[model.__ast__.body[0]].retty
    if isinstance(fwd_ret_ty, TyTensor):
        fwd_ret = 1
    elif isinstance(fwd_ret_ty, TyTuple):
        fwd_ret = fwd_ret_ty.size()
    else:
        raise NotImplementedError
    return node, node2type, fwd_ret


def grad_and_inject(model, inp, node2type, device):
    fwdbwd, attrs_order = grad_model(model, node2type, device)

    node2type_fwd, node2type_bwd = type_inference_fwdbwd(
        fwdbwd, (model, *inp), model)
    # print_inference_results(node2type_fwd)
    # print_inference_results(node2type_bwd)

    fill_shape_func(fwdbwd.body[0], node2type_fwd) # unbroadcast to reduce_sum
    fill_shape_func(fwdbwd.body[1], node2type_bwd) # unbroadcast to reduce_sum

    node2type_fwd, node2type_bwd = type_inference_fwdbwd(
        fwdbwd, (model, *inp), model)

    apply_passes_recursive(gast.Module(
        body=[fwdbwd.body[1]], type_ignores=[]), node2type, set())
    apply_passes_recursive(gast.Module(
        body=[fwdbwd.body[0]], type_ignores=[]), node2type, set())

    zero_fold(fwdbwd) # 0+x -> x
    advance_dce(fwdbwd)
    SplitToIndex().visit(fwdbwd)
    # invariant(fwdbwd)
    # exit(0)

    inject_fwdbwd(model, fwdbwd)


    self_attr_names = list(nd.id for nd in fwdbwd.body[1].args.args if anno.hasanno(nd, 'attr_name'))

    return fwdbwd, attrs_order, self_attr_names


def run_injected(model, fwdbwd, inp, device, fwd_ret):
    self_attrs = tuple(getattr(model, anno.getanno(nd, 'attr_name'))
                       for nd in fwdbwd.body[1].args.args if anno.hasanno(nd, 'attr_name'))
    with torch.no_grad():
        results = model.grad_forward(*inp)
        # print("[forward output]")
        # for rr in results:
        #     print("**", rr)
        args = (torch.ones([1], dtype=torch.float,
                device=device),) + results[fwd_ret:] + self_attrs
        gd = model.grad_backward(*args)
        # print("[backward output]")
        # for g in gd:
        #     print("**", g)


def test_sct_equal(model, inp, device, fwd_ret, optimizer, fwdbwd):
    self_attrs = tuple(getattr(model, anno.getanno(nd, 'attr_name'))
                       for nd in fwdbwd.body[1].args.args if anno.hasanno(nd, 'attr_name'))
    n_warmup = 100
    n_run = 100
    print("[SCT]")
    print("[warmup]")
    torch.cuda.synchronize()
    with torch.no_grad():
        for i in range(n_warmup):
            t0 = time()
            results = model.grad_forward(*inp)
            torch.cuda.synchronize()
            t1 = time()
            optimizer.zero_grad()
            torch.cuda.synchronize()
            t2 = time()
            args = (torch.ones([1], dtype=torch.float,
                                device=device),) + results[fwd_ret:] + self_attrs
            gd = model.grad_backward(*args)
            torch.cuda.synchronize()
            t3 = time()
            print("Time fwd = {} ms zero = {} ms bwd = {} ms sum = {} ms".format(
                (t1 - t0) * 1000, (t2 - t1) *
                1000, (t3 - t2) * 1000, (t3 - t0) * 1000
            ))
            # if i == 0: print(gd)

        timer = Timer("ms")
        torch.cuda.synchronize()
        for i in range(n_run):
            timer.start()
            results = model.grad_forward(*inp)
            optimizer.zero_grad()
            args = (torch.ones([1], dtype=torch.float,
                                device=device),) + results[fwd_ret:] + self_attrs
            model.grad_backward(*args)
            torch.cuda.synchronize()
            timer.log()
        timer.report()


def to_autograd(model, fwdbwd, node, inp, model_name, fwd_ret, attrs_order, self_attr_names):
    node2type_fwd, node2type_bwd = type_inference_fwdbwd(
        fwdbwd, (model, *inp), model)
    # attrs_order: order of forward inpt
    # self_attrs: order of ctx_save
    arg_nodes = to_torch_code(model_name, fwd_ret, fwdbwd, attrs_order, self_attr_names, node2type_fwd, node2type_bwd)
    model.forward, node, func_name, to_import = inject_training(model, node, arg_nodes, model_name)
    scope, func_name, model_dir = to_import
    torch_func = importlib.import_module(model_dir)
    setattr(scope, func_name, torch_func.GenTrainingModel)
    return node, func_name


def check_autograd(model, inp, optimizer, reference):
    output = model.forward(*inp)
    # print("[output-ORT]", output)
    optimizer.zero_grad()
    output.backward()
    # for name, param in model.named_parameters():
    #     if param.requires_grad:
    #         print(name, param.grad)
    for name, param in model.named_parameters():
        if param.requires_grad:
            close = torch.isclose(
                param.grad, reference[name], rtol=1e-3, atol=1e-3)
            if torch.any(~close):
                print(torch.masked_select(param.grad, ~close))
                print(torch.masked_select(reference[name], ~close))
                print(torch.masked_select(reference[name], ~close).shape)
                print(close)
            # assert(torch.allclose(
            #     param.grad, reference[name], rtol=1e-3, atol=1e-3))

    print("grad match!")


def test_performance(model, inp, optimizer, test_step=True):
    n_warmup = 100
    n_run = 100
    print("[warmup]")
    torch.cuda.synchronize()
    for i in range(n_warmup):
        t0 = time()
        output = model.forward(*inp)
        torch.cuda.synchronize()
        t1 = time()
        optimizer.zero_grad()
        torch.cuda.synchronize()
        t2 = time()
        output.backward()
        torch.cuda.synchronize()
        t3 = time()
        print("Time fwd = {} ms zero = {} ms bwd = {} ms sum = {} ms".format(
            (t1 - t0) * 1000, (t2 - t1) *
            1000, (t3 - t2) * 1000, (t3 - t0) * 1000
        ))

    timer = Timer("ms")
    torch.cuda.synchronize()
    for i in range(n_run):
        timer.start()
        output = model.forward(*inp)
        optimizer.zero_grad()
        output.backward()
        torch.cuda.synchronize()
        timer.log()
    print("[Training time]")
    timer.report()

    torch.cuda.synchronize()
    if test_step:
        timer_fwd = Timer("ms")
        timer_zero_grad = Timer("ms")
        timer_bwd = Timer("ms")
        for i in range(n_run):
            torch.cuda.synchronize()
            timer_fwd.start()
            output = model.forward(*inp)
            torch.cuda.synchronize()
            timer_fwd.log()
            timer_zero_grad.start()
            optimizer.zero_grad()
            torch.cuda.synchronize()
            timer_zero_grad.log()
            timer_bwd.start()
            output.backward()
            torch.cuda.synchronize()
            timer_bwd.log()
        print("[test_step]")
        timer_fwd.report()
        timer_zero_grad.report()
        timer_bwd.report()


def workflow_train_recursion(model, inp, model_name, device, profile=False, test_sct=False, test_step=False, use_nnfusion=True):
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    loss, reference = run_reference(model, inp, optimizer)
    node, node2type, fwd_ret = get_opt_ast_and_type(model, inp)
    fwdbwd, attrs_order, self_attr_names = grad_and_inject(model, inp, node2type, device)
    print("[after SCT]", astunparse.unparse(fwdbwd))

    run_injected(model, fwdbwd, inp, device, fwd_ret)
    if test_sct:
        test_sct_equal(model, inp, device, fwd_ret, optimizer, fwdbwd)
        exit(0)

    node, func_name = to_autograd(model, fwdbwd, node, inp, model_name, fwd_ret, attrs_order, self_attr_names)
    check_autograd(model, inp, optimizer, reference)

    node_fwd, node_bwd, node2type_fwd, node2type_bwd, inst = type_inference_fwdbwd_function(
        model, (model, *inp), node, func_name)

    buttom_up_feed.buttom_up_feed_train(node_fwd, node_bwd, model_name, node2type_fwd, node2type_bwd, inst, fwd_ret, attrs_order, func_name, model)
    check_autograd(model, inp, optimizer, reference)

    if not use_nnfusion: return
    test_performance(model, inp, optimizer)

import argparse
import ctypes
import os
import os.path as osp
import time

import numpy as np
import torch
import torch_blade
from model.pytorch import *

cuda = ctypes.CDLL("libcudart.so")

def tofp16model(in_file_name, out_file_name):
    from onnx import checker, load_model, save_model
    from onnxconverter_common import convert_float_to_float16
    onnx_model = load_model(in_file_name)
    trans_model = convert_float_to_float16(onnx_model, keep_io_types=False)
    checker.check_model(trans_model)
    save_model(trans_model, out_file_name)

def torch2onnx(prefix, model, inputs, fp16):
    outputs = model(*inputs)
    if not isinstance(outputs, (tuple, list)):
        outputs = (outputs, )
    input_names = ["input"+str(i) for i in range(len(inputs))]
    output_names = ["output"+str(i) for i in range(len(outputs))]
    torch.onnx.export(
        model, inputs,
        osp.join(prefix, "model.onnx"),
        input_names=input_names,
        output_names=output_names,
        export_params=True,
        training=torch.onnx.TrainingMode.EVAL,
        do_constant_folding=False,
        opset_version=11)
    if fp16:
        tofp16model( osp.join(prefix, "model.onnx"),  osp.join(prefix, "model.onnx"))
    feed_dict = dict(zip(input_names, inputs))
    np.savez(osp.join(prefix, "inputs.npz"), **feed_dict)

def run_blade(model, inputs):
    cu_inputs = []
    for item in inputs:
        cu_inputs.append(item.cuda() if isinstance(item, torch.Tensor) else item)

    torch_config = torch_blade.config.Config()
    torch_config.enable_mlir_amp = False # disable mix-precision
    model = torch.jit.trace(model, inputs, strict=False).cuda().eval()

    torch._C._jit_pass_inline(model._c.forward.graph)
    torch._C._jit_pass_remove_dropout(model._c)

    with torch.no_grad(), torch_config:
        # BladeDISC torch_blade optimize will return an optimized TorchScript
        model = torch_blade.optimize(model, allow_tracing=True, model_inputs=tuple(cu_inputs))

    def get_runtime():
        tic = time.time()
        _ = model(*cu_inputs)
        cuda.cudaDeviceSynchronize()
        return (time.time() - tic) * 1000
    with torch.no_grad():
        _ = [get_runtime() for i in range(50)] # warmup
        times = [get_runtime() for i in range(100)]
    print("mean: {}ms min: {}ms max: {}ms".format(np.mean(times), np.min(times), np.max(times)))
    # cuda.cudaProfilerStart()
    # get_runtime()
    # cuda.cudaProfilerStop()


def run_blade_trt(model, inputs):
    model = model.cuda().eval()
    cu_inputs = []
    for item in inputs:
        cu_inputs.append(item.cuda() if isinstance(item, torch.Tensor) else item)

    cfg = torch_blade.Config.get_current_context_or_new().clone()
    cfg.optimization_pipeline = torch_blade.tensorrt.backend_name()
    cfg.customize_onnx_opset_version = 12
    cfg.enable_fp16 = args.fp16
    model = torch.jit.trace(model, cu_inputs, strict=False).cuda().eval()
    with cfg, torch_blade.logging.logger_level_context('INFO'):
        model = torch_blade.optimization._static_optimize(model, False, model_inputs=tuple(cu_inputs))
    def get_runtime():
        tic = time.time()
        _ = model(*cu_inputs)
        cuda.cudaDeviceSynchronize()
        return (time.time() - tic) * 1000
    with torch.no_grad():
        _ = [get_runtime() for i in range(50)] # warmup
        times = [get_runtime() for i in range(100)]
    print("mean: {}ms min: {}ms max: {}ms".format(np.mean(times), np.min(times), np.max(times)))
    cuda.cudaProfilerStart()
    get_runtime()
    cuda.cudaProfilerStop()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=str)
    parser.add_argument("--bs", type=int, default=1)
    parser.add_argument("--prefix", type=str, default="temp")
    parser.add_argument("--fp16", action="store_true", default=False)
    parser.add_argument("--run_torch", action="store_true", default=False)
    args = parser.parse_args()
    assert (args.model in globals()), "Model {} not found.".format(args.model)

    torch.random.manual_seed(0)
    model, inputs = globals()[args.model](args.bs)

    if args.run_torch:
        if args.fp16:
            model = model.half()
            inputs = [x.half() if torch.is_floating_point(x) else x for x in inputs]
        run_blade(model, inputs)
    else:
        os.makedirs(args.prefix, exist_ok=True)
        torch2onnx(args.prefix, model, inputs, args.fp16)

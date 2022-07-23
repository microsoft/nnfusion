import torch
import numpy as np
import os.path as osp
import os
import argparse
import time

from model.pytorch import *

def tofp16model(in_file_name, out_file_name):
    from onnx import load_model, save_model
    from onnxmltools.utils import float16_converter
    onnx_model = load_model(in_file_name)
    trans_model = float16_converter.convert_float_to_float16(onnx_model,keep_io_types=True)
    save_model(trans_model, out_file_name)

def torch2onnx(prefix, model, inputs):
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
        training=False,
        do_constant_folding=False,
        opset_version=11)
    # tofp16model( osp.join(prefix, "model.onnx"),  osp.join(prefix, "model_fp16.onnx"))
    feed_dict = dict(zip(input_names, inputs))
    np.savez(osp.join(prefix, "inputs.npz"), **feed_dict)

def run_torch(model, inputs):
    model = model.cuda()
    model.eval()
    cu_inputs = []
    for item in inputs:
        cu_inputs.append(item.cuda() if isinstance(item, torch.Tensor) else item)
    def get_runtime():
        tic = time.time()
        _ = model(*cu_inputs)
        torch.cuda.synchronize()
        return (time.time() - tic) * 1000
    with torch.no_grad():
        _ = [get_runtime() for i in range(50)] # warmup
        times = [get_runtime() for i in range(100)]
    print("mean: {}ms min: {}ms max: {}ms".format(np.mean(times), np.min(times), np.max(times)))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=str)
    parser.add_argument("--bs", type=int, default=1)
    parser.add_argument("--prefix", type=str, default="temp")
    parser.add_argument("--run_torch", action="store_true", default=False)
    args = parser.parse_args()
    assert (args.model in globals()), "Model {} not found.".format(args.model)

    torch.random.manual_seed(0)
    model, inputs = globals()[args.model](args.bs)

    if args.run_torch:
        run_torch(model, inputs)
    else:
        os.makedirs(args.prefix, exist_ok=True)
        torch2onnx(args.prefix, model, inputs)

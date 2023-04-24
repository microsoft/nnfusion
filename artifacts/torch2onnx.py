import argparse
import os
import os.path as osp

import numpy as np
import torch
from model.pytorch import *


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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=str)
    parser.add_argument("--bs", type=int, default=1)
    parser.add_argument("--prefix", type=str, default="temp")
    parser.add_argument("--fp16", action="store_true", default=False)
    args = parser.parse_args()
    assert (args.model in globals()), "Model {} not found.".format(args.model)

    torch.random.manual_seed(0)
    model, inputs = globals()[args.model](args.bs)

    os.makedirs(args.prefix, exist_ok=True)
    torch2onnx(args.prefix, model, inputs, args.fp16)

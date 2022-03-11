import torch
import numpy as np
import os.path as osp
import os
import argparse
import time
from model.pytorch import *

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
        opset_version=9)
    feed_dict = dict(zip(input_names, inputs))
    np.savez(osp.join(prefix, "inputs.npz"), **feed_dict)
    with open(osp.join(prefix, "output_names.txt"), "w") as f:
        print(str(output_names), file=f)

def onnx2pb(prefix):
    import onnx
    from onnx_tf.backend import prepare
    onnx_model = onnx.load(osp.join(prefix, "model.onnx"))
    tf_rep = prepare(onnx_model, device="cuda")
    tf_rep.export_graph(osp.join(prefix, "model.pb"))

def run_torch(model, inputs):
    model = model.cuda()
    model.eval()
    def get_runtime():
        torch.cuda.synchronize()
        tic = time.time()
        cu_inputs = []
        for item in inputs:
            cu_inputs.append(item.cuda() if isinstance(item, torch.Tensor) else item)
        with torch.no_grad():
            _ = model(*cu_inputs)
        torch.cuda.synchronize()
        return time.time() - tic
    _ = [get_runtime() for i in range(50)] # warmup
    times = [get_runtime() for i in range(30)]
    print(np.mean(times), np.min(times), np.max(times))

if __name__ == "__main__":
    torch.random.manual_seed(0)
    prefix="temp"
    model, inputs = bert()
    os.makedirs(prefix, exist_ok=True)
    # torch2onnx(prefix, model, inputs)
    # onnx2pb(prefix)
    run_torch(model, inputs)

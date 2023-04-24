import tensorrt as trt
import numpy as np
import os.path as osp
import time
import argparse
import torch

def run_trt(prefix, use_fp16=False):
    logger = trt.Logger(trt.Logger.ERROR)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)
    with open(osp.join(prefix, "model.onnx"), 'rb') as f:
        success = parser.parse(f.read())
    if not success:
        for idx in range(parser.num_errors):
            print(parser.get_error(idx))
        raise RuntimeError()
    config = builder.create_builder_config()
    config.set_flag(trt.BuilderFlag.STRICT_TYPES)
    if use_fp16:
        config.set_flag(trt.BuilderFlag.FP16)
    engine = builder.build_engine(network, config)
    print("Built engine successfully.")

    tensors = []
    for binding in engine:
        shape = engine.get_binding_shape(binding)
        dtype = trt.nptype(engine.get_binding_dtype(binding))
        torch_dtype = torch.from_numpy(np.array([]).astype(dtype)).dtype
        if dtype == torch.float or dtype == torch.half:
            tensor = torch.randn(tuple(shape))
        else:
            tensor = torch.ones(tuple(shape))
        tensors.append(tensor.to(torch_dtype).cuda())

    context = engine.create_execution_context()
    buffer = [tensor.data_ptr() for tensor in tensors]
    def get_runtime():
        tic = time.monotonic_ns()
        context.execute_v2(buffer)
        return (time.monotonic_ns() - tic) / 1e6

    print("Warming up ...")
    st = time.time()
    while time.time() - st < 1.0:
        get_runtime() # warmup

    times = [get_runtime() for i in range(100)]
    print(f"avg: {np.mean(times)} ms")
    print(f"min: {np.min(times)} ms")
    print(f"max: {np.max(times)} ms")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--prefix', type=str, default="temp")
    parser.add_argument("--fp16", action="store_true", default=False)
    args = parser.parse_args()
    torch.random.manual_seed(0)
    run_trt(args.prefix, args.fp16)

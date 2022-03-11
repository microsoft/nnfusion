import tensorrt as trt
import numpy as np
import os.path as osp
import time
import argparse
import torch

def run_trt(prefix):
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

    engine = builder.build_cuda_engine(network)

    tensors = []
    for index, binding in enumerate(engine):
        shape = engine.get_binding_shape(binding)
        size = trt.volume(shape)
        dtype = trt.nptype(engine.get_binding_dtype(binding))
        torch_dtype = torch.from_numpy(np.array([]).astype(dtype)).dtype
        tensors.append(torch.Tensor(*shape).to(torch_dtype).cuda())

    input_tensor = []
    feed_dict = dict(np.load(osp.join(prefix, "inputs.npz"), allow_pickle=True))
    for item in feed_dict.values():
        input_tensor.append(torch.from_numpy(item))

    context = engine.create_execution_context()
    def get_runtime():
        tic = time.time()
        for i, tensor in enumerate(input_tensor):
            tensors[i] = tensor.cuda()
        buffer = [tensor.data_ptr() for tensor in tensors]
        context.execute(1, buffer)
        return time.time() - tic
    _ = [get_runtime() for i in range(50)] # warmup
    times = [get_runtime() for i in range(30)]
    print(np.mean(times), np.min(times), np.max(times))
    # print(tensors[1])

if __name__ == "__main__":
    torch.random.manual_seed(0)
    prefix = "temp"
    run_trt(prefix)

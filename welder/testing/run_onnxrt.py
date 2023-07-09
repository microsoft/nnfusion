import argparse
import os
import time

import numpy as np
import onnx
import onnxruntime as ort


def ref_output(onnx_model_path, device):
    onnx_model = onnx.load(onnx_model_path)
    onnx.checker.check_model(onnx_model)
    onnx_model = onnx.shape_inference.infer_shapes(onnx_model)
    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    if 'ROCMExecutionProvider' in ort.get_available_providers():
        providers = ['ROCMExecutionProvider', 'CPUExecutionProvider']
    elif 'CUDAExecutionProvider' in ort.get_available_providers():
        providers = [
            ('CUDAExecutionProvider', {
                'device_id': device,
                'arena_extend_strategy': 'kNextPowerOfTwo',
                'gpu_mem_limit': 8 * 1024 * 1024 * 1024,
                'cudnn_conv_algo_search': 'EXHAUSTIVE',
                'do_copy_in_default_stream': True,
            })
        ]
    else:
        raise RuntimeError("No valid Provider")
    ort_session = ort.InferenceSession(onnx_model_path, providers=providers, sess_options=sess_options)
    io_binding = ort_session.io_binding()
    for value in ort_session.get_inputs():
        if value.type == 'tensor(int64)':
            tensor = np.ones(value.shape).astype(np.int64)
        elif value.type == 'tensor(float16)':
            tensor = np.random.normal(size=value.shape).astype(np.float16)
        elif value.type == 'tensor(float)':
            tensor = np.random.normal(size=value.shape).astype(np.float32)
        else:
            raise NotImplementedError(value.type)
        # this eliminates H2D copy for profiling
        io_binding.bind_cpu_input(value.name, tensor)
    outputs = ort_session.get_outputs()
    # this eliminates D2H copy for profiling
    for item in outputs:
        io_binding.bind_output(item.name, 'cuda', device)
    ort_session.run_with_iobinding(io_binding)
    def get_runtime():
        tic = time.monotonic_ns()
        ort_session.run_with_iobinding(io_binding)
        return (time.monotonic_ns() - tic) / 1e6
    _ = [get_runtime() for i in range(50)] # warmup
    times = [get_runtime() for i in range(100)]
    print(np.mean(times), np.min(times), np.max(times))
    # print output
    output_data = io_binding.copy_outputs_to_cpu()
    for i, item in enumerate(outputs):
        # convert list into np array
        data = np.array(output_data[i]).flatten()
        
        # print first 10 elements and the last 
        print(item.name, data[:10], '...' ,data[-1])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--prefix', type=str, default="temp")
    parser.add_argument('--device', type=int, default=0)
    args = parser.parse_args()

    prefix = args.prefix
    ref_output(os.path.join(prefix, "model.onnx"), args.device)

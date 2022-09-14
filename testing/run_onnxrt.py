import numpy as np
import onnx
import onnxruntime as ort
import os
import time
import argparse

def ref_output(onnx_model_path, device):
    onnx_model = onnx.load(onnx_model_path)
    # onnx.checker.check_model(onnx_model)
    onnx_model = onnx.shape_inference.infer_shapes(onnx_model)
    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    providers = [
        ('CUDAExecutionProvider', {
            'device_id': device,
            'arena_extend_strategy': 'kNextPowerOfTwo',
            'gpu_mem_limit': 8 * 1024 * 1024 * 1024,
            'cudnn_conv_algo_search': 'EXHAUSTIVE',
            'do_copy_in_default_stream': True,
        })
    ]
    ort_session = ort.InferenceSession(onnx_model_path, providers=providers, sess_options=sess_options)
    io_binding = ort_session.io_binding()
    for value in ort_session.get_inputs():
        if value.type.find("int64") >= 0:
            tensor = np.ones(value.shape).astype(np.int64)
        elif value.type.find("float") >= 0:
            tensor = np.random.normal(size=value.shape).astype(np.float32)
        else:
            raise NotImplementedError(value.type)
        io_binding.bind_cpu_input(value.name, tensor)
    outputs = ort_session.get_outputs()

    for item in outputs:
        io_binding.bind_output(item.name)
    ort_session.run_with_iobinding(io_binding)
    def get_runtime():
        tic = time.time()
        ort_session.run_with_iobinding(io_binding)
        return (time.time() - tic) * 1000
    _ = [get_runtime() for i in range(200)] # warmup
    times = [get_runtime() for i in range(800)]
    print(np.mean(times), np.min(times), np.max(times))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--prefix', type=str, default="temp")
    parser.add_argument('--device', type=int, default=0)
    args = parser.parse_args()

    prefix = args.prefix
    ref_output(os.path.join(prefix, "model.onnx"), args.device)

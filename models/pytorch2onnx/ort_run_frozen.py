# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import onnxruntime as ort
import numpy as np
import argparse
import os
import sys
import time
import onnx
try:
    from yaml import safe_load as load_func
except ImportError:
    from json import loads as load_func

parser = argparse.ArgumentParser()
parser.add_argument('--file', type=str, default='./frozen_graph.onnx', help='The file name of the frozen graph.')
parser.add_argument('--optimized_model_filepath', type=str, default='', help='The file name of the optimized frozen graph.')
parser.add_argument('--graph_optimization_level', type=str, default='ORT_ENABLE_ALL', help='ONNX Runtime graph optimization level.')
parser.add_argument('--symbolic_dims', type=load_func, default={}, help='The size of symbolic dimensions, provided by \'{"dim1_name": dim1, "dim2_name": dim2}\'')
parser.add_argument('--warmup', type=int, default=5, help='The number of warmup iterations.')
parser.add_argument('--iters', type=int, default=100, help='The number of execution iterations.')
parser.add_argument('--provider', type=str, default='CPUExecutionProvider', help='The backend provider.')
parser.add_argument('--logger_severity', type=int, default=2, help='onnxruntime.set_default_logger_severity.')
parser.add_argument('--device_only', action="store_true", default=False, help='count device time instead of end2end time')
args = parser.parse_args()

if not os.path.exists(args.file):
    parser.exit(1, 'The specified file does not exist: {}'.format(args.file))

try:
    onnx.checker.check_model(args.file)
except Exception as e:
    print(e)
else:
    print("ONNX model check passed!")

def get_numpy(tensor):
    # ONNX Data Types Doc: https://github.com/onnx/onnx/blob/master/docs/IR.md#standard-data-types
    # ONNX Data Types Code: https://github.com/onnx/onnx/blob/master/onnx/defs/data_type_utils.h
    # NumPy Data Types: https://numpy.org/doc/stable/user/basics.types.html
    def get_numpy_dtype(onnx_dtype):
        if 'float16' in onnx_dtype:
            return np.float16
        elif 'float' in onnx_dtype:
            return np.float32
        elif 'double' in onnx_dtype:
            return np.float64
        elif 'uint8' in onnx_dtype:
            return np.uint8
        elif 'uint16' in onnx_dtype:
            return np.uint16
        elif 'int8' in onnx_dtype:
            return np.int8
        elif 'int16' in onnx_dtype:
            return np.int16
        elif 'int32' in onnx_dtype:
            return np.int32
        elif 'int64' in onnx_dtype:
            return np.int64
        elif 'bool' in onnx_dtype:
            return np.bool_
        else:
            raise NotImplementedError(onnx_dtype + " is not supported in this script yet.")
        return np.float32

    def check_shape(shape):
        for dim in shape:
            if isinstance(dim, int):
                continue
            elif isinstance(dim, str):
                raise Exception(f"Unknown symbilic dimension: {dim}")
            else:
                raise Exception(f"Unknown dimension type: {type(dim)}")

    dtype = get_numpy_dtype(tensor.type)
    shape = tensor.shape
    check_shape(shape)
    return np.ones(shape, dtype=dtype)

# print("Execution Device:", ort.get_device())

print("Importing ONNX model into ONNX Runtime...")
ort.set_default_logger_severity(args.logger_severity)

sess_options = ort.SessionOptions()
sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
if args.graph_optimization_level == 'ORT_DISABLE_ALL':
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
elif args.graph_optimization_level == 'ORT_ENABLE_BASIC':
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
elif args.graph_optimization_level == 'ORT_ENABLE_EXTENDED':
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
if args.optimized_model_filepath != '':
    sess_options.optimized_model_filepath = args.optimized_model_filepath

for k, v in args.symbolic_dims.items():
    sess_options.add_free_dimension_override_by_name(k, int(v))

providers = args.provider.split(",")
if "CPUExecutionProvider" not in providers:
    providers.append("CPUExecutionProvider")
if 'CUDAExecutionProvider' in ort.get_available_providers() and 'CUDAExecutionProvider' not in providers:
    providers = ['CUDAExecutionProvider'] + providers
if 'ROCMExecutionProvider' in ort.get_available_providers() and 'ROCMExecutionProvider' not in providers:
    providers = ['ROCMExecutionProvider'] + providers

ort_session = ort.InferenceSession(args.file, sess_options, providers=providers)

print("Execution Providers:", ort_session.get_providers())

inputs = ort_session.get_inputs()
inputs_name = [item.name for item in inputs]
ort_inputs = {}
for tensor in inputs:
    ort_inputs.update({tensor.name: get_numpy(tensor)})

outputs = ort_session.get_outputs()
outputs_name = [item.name for item in outputs]

for warmup in range(args.warmup):
    outputs = ort_session.run(outputs_name, ort_inputs)
    for i in range(len(outputs)):
        out_flat = outputs[i].flat
        if (len(out_flat) > 0):
            max_len = min(10, len(out_flat))
            print(outputs_name[i])
            print(out_flat[:max_len], "...(size=", len(out_flat), "end with", out_flat[-1], ")")
            # print_offset = int(len(out_flat) / 3)
            # max_len = min(10, len(out_flat) - print_offset)
            # print(out_flat[print_offset:max_len + print_offset], "offset=", print_offset)

if args.device_only:
    io_binding = ort_session.io_binding()
    for key, value in ort_inputs.items():
        io_binding.bind_ortvalue_input(key, ort.OrtValue.ortvalue_from_numpy(value, 'cuda', 0))
    for o in outputs_name:
        io_binding.bind_output(o, 'cuda')

if args.iters > 0:
    print('>> Evaluating Benchmark ...')
    t_start = time.time()
    for step in range(args.iters):
        if args.device_only:
            ort_session.run_with_iobinding(io_binding)
        else:
            ort_session.run(outputs_name, ort_inputs)
    t_end = time.time()
    print('>> Average time for each run: %.4f ms;' % ((t_end - t_start) * 1e3 / args.iters))

# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import onnxruntime as ort
import numpy as np
import argparse
import os
import sys
import time
import onnx

parser = argparse.ArgumentParser()
parser.add_argument('--file', type=str, default='./frozen_graph.onnx', help='The file name of the frozen graph.')
parser.add_argument('--warmup', type=int, default=5, help='The number of warmup iterations.')
parser.add_argument('--iters', type=int, default=100, help='The number of execution iterations.')
parser.add_argument('--provider', type=str, default='', help='The backend provider.')
parser.add_argument('--logger_severity', type=int, default=2, help='onnxruntime.set_default_logger_severity.')
args = parser.parse_args()

if not os.path.exists(args.file):
    parser.exit(1, 'The specified file does not exist: {}'.format(args.file))

onnx.checker.check_model(args.file)
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
        elif 'int8' in onnx_dtype:
            return np.int8
        elif 'int16' in onnx_dtype:
            return np.int16
        elif 'int32' in onnx_dtype:
            return np.int32
        elif 'int64' in onnx_dtype:
            return np.int64
        elif 'uint8' in onnx_dtype:
            return np.uint8
        elif 'uint16' in onnx_dtype:
            return np.uint16
        elif 'bool' in onnx_dtype:
            return np.bool_
        else:
            raise NotImplementedError(onnx_dtype + " is not supported in this script yet.")
        return np.float32

    dtype = get_numpy_dtype(tensor.type)
    shape = tensor.shape
    return np.ones(shape, dtype=dtype)

# print("Execution Device:", ort.get_device())

print("Importing ONNX model into ONNX Runtime...")
ort.set_default_logger_severity(args.logger_severity)
sess_options = ort.SessionOptions()
sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
ort_session = ort.InferenceSession(args.file, sess_options)

if args.provider != '':
    ort_session.set_providers([args.provider])

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

print('>> Evalutating Benchmark ...')
t_start = time.time()
for step in range(args.iters):
    ort_session.run(outputs_name, ort_inputs)
t_end = time.time()
print('>> Average time for each run: %.4f ms;' % ((t_end - t_start) * 1e3 / args.iters))

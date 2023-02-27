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
parser.add_argument('--graph_optimization_level', type=str, default='ORT_ENABLE_ALL', help='ONNX Runtime graph optimization level.')
parser.add_argument('--symbolic_dims', type=load_func, default={}, help='The size of symbolic dimensions, provided by \'{"dim1_name": dim1, "dim2_name": dim2}\'')
parser.add_argument('--provider', type=str, default='CPUExecutionProvider', help='The backend provider.')
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

model = onnx.load(args.file)
node_name_dict = {}
for node_id in range(len(model.graph.node)):
    node_outputs = model.graph.node[node_id].output
    for output_id in range(len(node_outputs)):
        debug_tensor = onnx.helper.ValueInfoProto()
        debug_tensor.name = node_outputs[output_id]
        model.graph.output.append(debug_tensor)
        node_name_dict.update({debug_tensor.name: model.graph.node[node_id].name+'_'+str(output_id)})

node_inits = model.graph.initializer
for init_id in range(len(node_inits)):
    debug_tensor = onnx.helper.ValueInfoProto()
    debug_tensor.name = node_inits[init_id].name
    model.graph.output.append(debug_tensor)

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

for k, v in args.symbolic_dims.items():
    sess_options.add_free_dimension_override_by_name(k, int(v))

providers = args.provider.split(",")
if "CPUExecutionProvider" not in providers:
    providers.append("CPUExecutionProvider")

ort_session = ort.InferenceSession(model.SerializeToString(), sess_options, providers=providers)

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

for step in range(1):
    outputs = ort_session.run(outputs_name, ort_inputs)
    for i in range(len(outputs)):
        out_flat = outputs[i].flat
        if (len(out_flat) > 0):

            max_len = min(10, len(out_flat))
            # if (outputs_name[i] == "onnx::MatMul_8354"):
            #     print (','.join([str(i) for i in out_flat]))

            print(outputs_name[i])
            print(out_flat[:max_len], f"...(size = {len(out_flat)} end with {out_flat[-1]}, sum = {np.sum(out_flat)})")
            # print_offset = int(len(out_flat) / 3)
            # max_len = min(10, len(out_flat) - print_offset)
            # print(out_flat[print_offset:max_len + print_offset], "offset=", print_offset)


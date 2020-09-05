import numpy as np
import tvm
from tvm import autotvm
import topi
import topi.testing
from tvm.contrib.pickle_memoize import memoize
from topi.util import get_const_tuple
import json
import os
import tensorflow as tf

flags = tf.flags
flags.DEFINE_string("input_path", "", "path of input file")
flags.DEFINE_string("tvm_profile_log",
                    "/tmp/tvm_profile.log", "path of tvm profile")
flags.DEFINE_string("output_path", "", "path of output file")

FLAGS = flags.FLAGS


def topi_conv2d_nchw_codegen(input_shape, filter_shape, output_shape, strides, paddings, dilations, add_bias=False, add_relu=False):
    input_n, input_c, input_h, input_w = input_shape
    filter_c, filter_input_c, filter_h, filter_w = filter_shape
    output_n, output_c, output_h, output_w = output_shape
    stride_h, stride_w = strides
    padding_h, padding_w = paddings
    dilation_h, dilation_w = dilations

    A = tvm.placeholder(input_shape, name='input0')
    W = tvm.placeholder(filter_shape, name='input1')
    bias = tvm.placeholder((filter_c, 1, 1), name='input2')

    a_shape = get_const_tuple(A.shape)
    w_shape = get_const_tuple(W.shape)
    bias_shape = get_const_tuple(bias.shape)
    dtype = A.dtype
    device = 'cuda'

    ctx = tvm.context(device, 0)

    a_np = np.random.uniform(size=a_shape).astype(dtype)
    w_np = np.random.uniform(size=w_shape).astype(dtype)
    b_np = np.random.uniform(size=bias_shape).astype(dtype)
    c_np = np.zeros(output_shape).astype("float32")

    with tvm.target.create(device):
        C = topi.nn.conv2d(A, W, strides, paddings,
                           dilations, layout='NCHW', out_dtype=dtype)
        if add_bias:
            C = topi.add(C, bias)
        if add_relu:
            C = topi.nn.relu(C)
        s = topi.generic.schedule_conv2d_nchw([C])

    a = tvm.nd.array(a_np, ctx)
    w = tvm.nd.array(w_np, ctx)
    b = tvm.nd.array(b_np, ctx)
    c = tvm.nd.array(c_np, ctx)

    if add_bias:
        func = tvm.build(s, [A, W, bias, C], device, name="topi_convolution_op_float_i%d_%d_%d_%d_w%d_%d_%d_%d_o%d_%d_%d_%d_ws%d_%d_wd%d_%d_p%d_%d" % (
            input_n, input_c, input_h, input_w, filter_c, filter_input_c, filter_h, filter_w, output_n, output_c, output_h, output_w, stride_h, stride_w, dilation_h, dilation_w, padding_h, padding_w))
        func(a, w, b, c)
    else:
        func = tvm.build(s, [A, W, C], device, name="topi_convolution_op_float_i%d_%d_%d_%d_w%d_%d_%d_%d_o%d_%d_%d_%d_ws%d_%d_wd%d_%d_p%d_%d" % (
            input_n, input_c, input_h, input_w, filter_c, filter_input_c, filter_h, filter_w, output_n, output_c, output_h, output_w, stride_h, stride_w, dilation_h, dilation_w, padding_h, padding_w))
        func(a, w, c)

    return func.imported_modules[0].get_source()

# cudnn_convolution_op_float|i+1_3_32_32|w+64_3_3_3|o+1_64_32_32|ws+1_1|wd+1_1|p+1_1


def extract_ops_from_log(log_path):
    conv_ops = []
    lines = open(log_path).readlines()
    deduped_lines = list(set(lines))
    print("#convs:", len(lines), "#deduped_convs:", len(deduped_lines))
    for line in deduped_lines:
        items = line.rstrip('\n').split('|')

        tmp_items = items[1].split('+')[1].split('_')
        input_shape = [int(item) for item in tmp_items]

        tmp_items = items[2].split('+')[1].split('_')
        filter_shape = [int(item) for item in tmp_items]

        tmp_items = items[3].split('+')[1].split('_')
        output_shape = [int(item) for item in tmp_items]

        tmp_items = items[4].split('+')[1].split('_')
        window_movement_strides = [int(item) for item in tmp_items]

        tmp_items = items[5].split('+')[1].split('_')
        window_dilation_strides = [int(item) for item in tmp_items]

        tmp_items = items[6].split('+')[1].split('_')
        padding_below_diff = [int(item) for item in tmp_items]

        conv_params = {
            'input_shape': input_shape,
            'filter_shape': filter_shape,
            'output_shape': output_shape,
            'window_movement_strides': window_movement_strides,
            'window_dilation_strides': window_dilation_strides,
            'padding_below_diff': padding_below_diff,
        }
        conv_ops.append(conv_params)

    return conv_ops


def get_tvm_topi_func_name(input_shape, filter_shape, output_shape, strides, paddings, dilations):
    input_n, input_c, input_h, input_w = input_shape
    filter_c, filter_input_c, filter_h, filter_w = filter_shape
    output_n, output_c, output_h, output_w = output_shape
    stride_h, stride_w = strides
    padding_h, padding_w = paddings
    dilation_h, dilation_w = dilations
    func_name = "topi_convolution_op_float_i%d_%d_%d_%d_w%d_%d_%d_%d_o%d_%d_%d_%d_ws%d_%d_wd%d_%d_p%d_%d_kernel0" % (
        input_n, input_c, input_h, input_w, filter_c, filter_input_c, filter_h, filter_w, output_n, output_c, output_h, output_w, stride_h, stride_w, dilation_h, dilation_w, padding_h, padding_w)
    return func_name


def extract_tvm_profiling_from_log(log_path):
    lines = open(log_path).readlines()
    deduped_lines = list(set(lines))
    # print("#convs:", len(lines), "#deduped_convs:", len(deduped_lines))
    profiling_result = {}
    for line in deduped_lines:
        items = line.rstrip('\n').split('|')
        profiling_data = {
            'gridDim': [int(items[1]), int(items[2]), int(items[3])],
            'blockDim': [int(items[4]), int(items[5]), int(items[6])]
        }
        profiling_result.update({items[0]: profiling_data})
    return profiling_result


def generate_db_topi_ops(conv_ops):
    topi_ops = []
    tvm_profiling_log_path = FLAGS.tvm_profile_log
    if os.path.exists(tvm_profiling_log_path):
        os.remove(tvm_profiling_log_path)

    for conv_op in conv_ops:
        topi_code = topi_conv2d_nchw_codegen(conv_op['input_shape'], conv_op['filter_shape'], conv_op['output_shape'],
                                             conv_op['window_movement_strides'], conv_op['padding_below_diff'], conv_op['window_dilation_strides'])
        topi_op = {
            'tvm_func_name': get_tvm_topi_func_name(conv_op['input_shape'], conv_op['filter_shape'], conv_op['output_shape'],
                                                    conv_op['window_movement_strides'], conv_op['padding_below_diff'], conv_op['window_dilation_strides']),
            'op_type': 'Convolution',
            'parameters': conv_op,
            'code': topi_code
        }
        topi_ops.append(topi_op)

    profiling_result = extract_tvm_profiling_from_log(tvm_profiling_log_path)
    for topi_op in topi_ops:
        tvm_func_name = topi_op['tvm_func_name']
        topi_op.update(profiling_result[tvm_func_name])

    return topi_ops


conv_ops = extract_ops_from_log(FLAGS.input_path)
topi_ops = generate_db_topi_ops(conv_ops)

with open(FLAGS.output_path, 'w') as fout:
    json.dump(topi_ops, fout)

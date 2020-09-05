import numpy as np
import tvm
from tvm import autotvm
import topi
import logging
import topi.testing
from tvm.contrib.pickle_memoize import memoize
from topi.util import get_const_tuple
import json
import os
import sys

@autotvm.template
def tvm_conv2d_nchw_tune_op(input_shape, filter_shape, output_shape, strides, paddings, dilations):
    A = tvm.placeholder(input_shape, name='input0', dtype="float32")
    W = tvm.placeholder(filter_shape, name='input1', dtype="float32")
    C = topi.nn.conv2d_nchw(A, W, strides, paddings, dilations, out_dtype="float32")
    cfg = autotvm.get_config()
    s = topi.cuda.conv2d.schedule_conv2d_nchw_cuda(cfg, C)
    return s, [A, W, C]

def search_conv2d_nchw_configs(input_shape, filter_shape, output_shape, strides, paddings, dilations, num_trials):
    input_n, input_c, input_h, input_w = input_shape
    filter_c, filter_input_c, filter_h, filter_w = filter_shape
    output_n, output_c, output_h, output_w = output_shape
    stride_h, stride_w = strides
    padding_h, padding_w = paddings
    dilation_h, dilation_w = dilations

    logging.getLogger('autotvm').setLevel(logging.DEBUG)
    logging.getLogger('autotvm').addHandler(logging.StreamHandler(sys.stdout))
    task = autotvm.task.create(tvm_conv2d_nchw_tune_op, args=(input_shape, filter_shape, output_shape, strides, paddings, dilations), target='cuda')
    print(task.config_space)
    measure_option = autotvm.measure_option(
        builder=autotvm.LocalBuilder(),
        runner=autotvm.LocalRunner(repeat=3, min_repeat_ms=100, timeout=4)
    )

    op_name="tuned_convolution_op_float_i%d_%d_%d_%d_w%d_%d_%d_%d_o%d_%d_%d_%d_ws%d_%d_wd%d_%d_p%d_%d" % (input_n, input_c, input_h, input_w, filter_c, filter_input_c, filter_h, filter_w, output_n, output_c, output_h, output_w, stride_h, stride_w, dilation_h, dilation_w, padding_h, padding_w)

    log_name = "tuned_kernels/"+op_name+".log"

    tuner = autotvm.tuner.XGBTuner(task)
    tuner.tune(n_trial=num_trials, measure_option=measure_option,
               callbacks=[autotvm.callback.log_to_file(log_name)])


def extract_ops_from_log(log_path):
    conv_ops = []
    lines = open(log_path).readlines()
    deduped_lines = list(set(lines))
    print("#convs:", len(lines), "#deduped_convs:", len(deduped_lines))
    for line in deduped_lines:
        # print(line.rstrip('\n'))
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
    # print(deduped_lines)
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
    tvm_profiling_log_path = '/home/jxue/vlima/projects/blockfusion-model/kernels/tvm_profile.log'
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


conv_ops = extract_ops_from_log("nasnet_large_conv_kernels.txt")
for conv_op in conv_ops:
    search_conv2d_nchw_configs(conv_op['input_shape'], conv_op['filter_shape'], conv_op['output_shape'],
                               conv_op['window_movement_strides'], conv_op['padding_below_diff'], conv_op['window_dilation_strides'], num_trials=2000)
# topi_ops = generate_db_topi_ops(conv_ops)


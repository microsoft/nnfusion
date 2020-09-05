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

# @autotvm.template
def tvm_depthwise_conv2d_nchw_tune_op(input_shape, filter_shape, strides, paddings, dilations):
    A = tvm.placeholder(input_shape, name='input0', dtype="float32")
    W = tvm.placeholder(filter_shape, name='input1', dtype="float32")
    C = topi.nn.depthwise_conv2d_nchw(A, W, strides, paddings, dilations, out_dtype="float32")
    cfg = autotvm.get_config()
    s = topi.cuda.depthwise_conv2d.schedule_depthwise_conv2d_nchw_cuda(cfg, C)
    return s, [A, W, C]

def search_depthwise_conv2d_nchw_configs(input_shape, filter_shape, strides, paddings, dilations, num_trials):
    input_n, input_c, input_h, input_w = input_shape
    filter_c, filter_input_c, filter_h, filter_w = filter_shape
    # output_n, output_c, output_h, output_w = output_shape
    stride_h, stride_w = strides
    padding_h, padding_h, padding_w, padding_w = paddings
    dilation_h, dilation_w = dilations

    logging.getLogger('autotvm').setLevel(logging.DEBUG)
    logging.getLogger('autotvm').addHandler(logging.StreamHandler(sys.stdout))
    task = autotvm.task.create(tvm_depthwise_conv2d_nchw_tune_op, args=(input_shape, filter_shape, strides, paddings, dilations), target='cuda', template_key='direct')
    print(task.config_space)
    measure_option = autotvm.measure_option(
        builder=autotvm.LocalBuilder(),
        runner=autotvm.LocalRunner(repeat=3, min_repeat_ms=100, timeout=4)
    )

    op_name="tuned_depthwise_convolution_op_float_i%d_%d_%d_%d_w%d_%d_%d_%d_ws%d_%d_wd%d_%d_p%d_%d" % (input_n, input_c, input_h, input_w, filter_c, filter_input_c, filter_h, filter_w, stride_h, stride_w, dilation_h, dilation_w, padding_h, padding_w)

    log_name = "tuned_depthwise_convolution_kernels/"+op_name+".log"

    tuner = autotvm.tuner.XGBTuner(task)
    tuner.tune(n_trial=num_trials, measure_option=measure_option,
               callbacks=[autotvm.callback.log_to_file(log_name)])


search_depthwise_conv2d_nchw_configs((1, 32, 32, 32), (32, 1, 5, 5), (1, 1), (2, 2, 2, 2), (1, 1), 1000)
search_depthwise_conv2d_nchw_configs((1, 32, 32, 32), (32, 1, 3, 3), (1, 1), (1, 1, 1, 1), (1, 1), 1000)
search_depthwise_conv2d_nchw_configs((1, 96, 32, 32), (96, 1, 3, 3), (1, 1), (1, 1, 1, 1), (1, 1), 1000)
search_depthwise_conv2d_nchw_configs((1, 96, 32, 32), (96, 1, 5, 5), (1, 1), (2, 2, 2, 2), (1, 1), 1000)
search_depthwise_conv2d_nchw_configs((1, 64, 16, 16), (64, 1, 7, 7), (1, 1), (3, 3, 3, 3), (1, 1), 1000)
search_depthwise_conv2d_nchw_configs((1, 64, 32, 32), (64, 1, 7, 7), (2, 2), (2, 2, 3, 3), (1, 1), 1000)
search_depthwise_conv2d_nchw_configs((1, 64, 16, 16), (64, 1, 5, 5), (1, 1), (2, 2, 2, 2), (1, 1), 1000)
search_depthwise_conv2d_nchw_configs((1, 64, 32, 32), (64, 1, 5, 5), (2, 2), (1, 1, 2, 2), (1, 1), 1000)
search_depthwise_conv2d_nchw_configs((1, 64, 16, 16), (64, 1, 3, 3), (1, 1), (1, 1, 1, 1), (1, 1), 1000)
search_depthwise_conv2d_nchw_configs((1, 128, 8, 8), (128, 1, 7, 7), (1, 1), (3, 3, 3, 3), (1, 1), 1000)
search_depthwise_conv2d_nchw_configs((1, 128, 16, 16), (128, 1, 7, 7), (2, 2), (2, 2, 3, 3), (1, 1), 1000)
search_depthwise_conv2d_nchw_configs((1, 128, 8, 8), (128, 1, 5, 5), (1, 1), (2, 2, 2, 2), (1, 1), 1000)
search_depthwise_conv2d_nchw_configs((1, 128, 16, 16), (128, 1, 5, 5), (2, 2), (1, 1, 2, 2), (1, 1), 1000)
search_depthwise_conv2d_nchw_configs((1, 128, 8, 8), (128, 1, 3, 3), (1, 1), (1, 1, 1, 1), (1, 1), 1000)



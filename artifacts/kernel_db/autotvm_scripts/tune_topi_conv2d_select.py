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
import tensorflow as tf

flags = tf.flags
flags.DEFINE_string("input_path", "", "path of input file")
flags.DEFINE_string("autotvm_log", "", "path of autotvm tuning log")
flags.DEFINE_string("tvm_profile_log",
                    "/tmp/tvm_profile.log", "path of tvm profile")
flags.DEFINE_string("output_path", "", "path of output file")

FLAGS = flags.FLAGS

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

    # if DO_TUNING:
    tuner = autotvm.tuner.XGBTuner(task)
    # set num of trial
    tuner.tune(n_trial=num_trials, measure_option=measure_option,
               callbacks=[autotvm.callback.log_to_file(log_name)])


def lookup_conv2d_config(input_shape, filter_shape, output_shape, strides, paddings, dilations, log_path):
    input_n, input_c, input_h, input_w = input_shape
    filter_c, filter_input_c, filter_h, filter_w = filter_shape
    output_n, output_c, output_h, output_w = output_shape
    stride_h, stride_w = strides
    padding_h, padding_w = paddings
    dilation_h, dilation_w = dilations

    op_name="tuned_convolution_op_float_i%d_%d_%d_%d_w%d_%d_%d_%d_o%d_%d_%d_%d_ws%d_%d_wd%d_%d_p%d_%d" % (input_n, input_c, input_h, input_w, filter_c, filter_input_c, filter_h, filter_w, output_n, output_c, output_h, output_w, stride_h, stride_w, dilation_h, dilation_w, padding_h, padding_w)

    log_name = FLAGS.autotvm_log
    with open(log_name, "r") as fin:
        log_lines = fin.readlines()
    # log_records=tvm.autotvm.record.load_from_file(log_name)
    log_records = []
    for line in log_lines:
        line = line.rstrip('\n')
        # print(line)
        record_json = json.loads(line)
        tm = record_json['r'][0][0]
        if tm > 10000000:  # filter bad configs
            continue
        if record_json['i'][2][0] != input_shape or record_json['i'][2][1] != filter_shape or record_json['i'][2][2] != output_shape or record_json['i'][2][3] != strides or record_json['i'][2][4] != paddings or record_json['i'][2][5] != dilations: # filter other configs
            continue
        griddim_x = record_json['i'][5]["e"][2][2][0]
        if griddim_x == -1:
            griddim_x = int(output_w / record_json['i'][5]["e"][2][2][1] / record_json['i'][5]["e"][2][2][2] / record_json['i'][5]["e"][2][2][3])
        griddim_y = record_json['i'][5]["e"][1][2][0]
        if griddim_y == -1:
            griddim_y = int(output_h / record_json['i'][5]["e"][1][2][1] / record_json['i'][5]["e"][1][2][2] / record_json['i'][5]["e"][1][2][3])
        griddim_z = record_json['i'][5]["e"][0][2][0]
        if griddim_z == -1:
            griddim_z = int(output_n * output_c / record_json['i'][5]["e"][0][2][1] / record_json['i'][5]["e"][0][2][2] / record_json['i'][5]["e"][0][2][3])
        record = {"time": tm,
                  "grid": [griddim_x, griddim_y, griddim_z],
                  "block": [record_json['i'][5]["e"][2][2][2], record_json['i'][5]["e"][1][2][2], record_json['i'][5]["e"][0][2][2]],
                  "config": line}
        # if record["block"][0] * record["block"][1] * record["block"][2] % 32 != 0:
        #     continue
        opt = tm * record["grid"][0] * record["grid"][1] * record["grid"][2] * record["block"][0] * record["block"][1] * record["block"][2]
        if record["block"][0] * record["block"][1] * record["block"][2] % 32 != 0:
            opt = tm * record["grid"][0] * record["grid"][1] * record["grid"][2] * (record["block"][0] * record["block"][1] * record["block"][2] / 32 + 1) * 32
        # opt = record["grid"][0] * record["grid"][1] * record["grid"][2] * record["block"][0] * record["block"][1] * record["block"][2]
        record.update({"opt": opt})
        log_records.append((tm, record))
        # print(log_records[-1])
    log_records.sort(key=lambda item: item[0])
    # print("available kernels:", len(log_records))
    print(op_name)
    log_records_fast = log_records[0:100] # top fast kernels
    # log_records_fast = log_records
    log_records = []
    for i in range(len(log_records_fast)):
        log_records.append((log_records_fast[i][1]["opt"], log_records_fast[i][1]))
    log_records.sort(key=lambda item: item[0])
    print("fastest kernel:",log_records_fast[0][1]["time"], "grid:", log_records_fast[0][1]["grid"], "block:", log_records_fast[0][1]["block"])
    print("efficient kernel:",log_records[0][1]["time"], "grid:", log_records[0][1]["grid"], "block:", log_records[0][1]["block"])
    # for i in range(min(10, len(log_records))): # print top 100 entries
    #     print("time:", log_records[i][1]["time"], "grid:", log_records[i][1]["grid"], "block:", log_records[i][1]["block"])
    #     print(log_records[i][1]["config"])
    with open(log_path, 'a') as fout:
        fout.write(log_records[0][1]["config"]+'\n')
        # json.dump(log_records[0][1]["config"], fout)

def tune_conv2d_nchw_codegen(input_shape, filter_shape, output_shape, strides, paddings, dilations):
    input_n, input_c, input_h, input_w = input_shape
    filter_c, filter_input_c, filter_h, filter_w = filter_shape
    output_n, output_c, output_h, output_w = output_shape
    stride_h, stride_w = strides
    padding_h, padding_w = paddings
    dilation_h, dilation_w = dilations

    lookup_conv2d_config(input_shape, filter_shape, output_shape, strides, paddings, dilations)

    logging.getLogger('autotvm').setLevel(logging.DEBUG)
    logging.getLogger('autotvm').addHandler(logging.StreamHandler(sys.stdout))
    task = autotvm.task.create(tvm_conv2d_nchw_tune_op, args=(input_shape, filter_shape, output_shape, strides, paddings, dilations), target='cuda')

    op_name="tuned_convolution_op_float_i%d_%d_%d_%d_w%d_%d_%d_%d_o%d_%d_%d_%d_ws%d_%d_wd%d_%d_p%d_%d" % (input_n, input_c, input_h, input_w, filter_c, filter_input_c, filter_h, filter_w, output_n, output_c, output_h, output_w, stride_h, stride_w, dilation_h, dilation_w, padding_h, padding_w)

    # log_name = "tuned_kernels/"+op_name+".log"
    log_name = "tmp.log"

    dispatch_context = autotvm.apply_history_best(log_name)
    best_config = dispatch_context.query(task.target, task.workload)

    with dispatch_context:
        with tvm.target.create('cuda'):
            s, arg_bufs = tvm_conv2d_nchw_tune_op(input_shape, filter_shape, output_shape, strides, paddings, dilations)
            func = tvm.build(s, arg_bufs, 'cuda', name=op_name)

    ctx = tvm.context('cuda', 0)

    a_np = np.random.uniform(size=input_shape).astype("float32")
    w_np = np.random.uniform(size=filter_shape).astype("float32")
    c_np = np.zeros(output_shape).astype("float32")

    a = tvm.nd.array(a_np, ctx)
    w = tvm.nd.array(w_np, ctx)
    c = tvm.nd.array(c_np, ctx)

    kernel_code = func.imported_modules[0].get_source()

    func(a, w, c)

    return kernel_code

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
    func_name = "tuned_convolution_op_float_i%d_%d_%d_%d_w%d_%d_%d_%d_o%d_%d_%d_%d_ws%d_%d_wd%d_%d_p%d_%d_kernel0" % (
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
    tvm_profiling_log_path = FLAGS.tvm_profile_log
    if os.path.exists(tvm_profiling_log_path):
        os.remove(tvm_profiling_log_path)

    for conv_op in conv_ops:
        topi_code = tune_conv2d_nchw_codegen(conv_op['input_shape'], conv_op['filter_shape'], conv_op['output_shape'],
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



output_log_file = FLAGS.output_path

if os.path.exists(output_log_file):
    os.remove(output_log_file)

conv_ops = extract_ops_from_log(FLAGS.input_path)
# conv_op = conv_ops[0]
# lookup_conv2d_config(conv_op['input_shape'], conv_op['filter_shape'], conv_op['output_shape'],
#                      conv_op['window_movement_strides'], conv_op['padding_below_diff'], conv_op['window_dilation_strides'])
# topi_ops = generate_db_topi_ops(conv_ops)


for conv_op in conv_ops:
    lookup_conv2d_config(conv_op['input_shape'], conv_op['filter_shape'], conv_op['output_shape'],
                                             conv_op['window_movement_strides'], conv_op['padding_below_diff'], conv_op['window_dilation_strides'], output_log_file)

# with open('resnext-select-efficient_convolution_kernels.json', 'w') as fout:
#     json.dump(topi_ops, fout)
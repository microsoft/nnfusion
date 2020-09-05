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
flags.DEFINE_string("profile_log",
                    "profile.log", "path of profile log")
flags.DEFINE_string("output_path", "../example_kernel_db.json", "path of output file")

FLAGS = flags.FLAGS

def manual_dot_codegen(m, k, n):
    op_name = "manual_dot_nn_op_float_m%d_k%d_n%d" % (m, k, n)

    code_file = op_name + ".cu"

    kernel_code = open(code_file).read()

    return kernel_code


def extract_ops_from_log():
    dot_ops = []
    dot_ops.append({'arg0_shape': [1, 256], 'arg1_shape': [256, 256], 'out_shape': [1, 256], 'transpose_A': False, 'transpose_B': False})
    return dot_ops


def get_tvm_topi_func_name(m, k, n):
    func_name = "manual_dot_nn_op_float_m%d_k%d_n%d_kernel0" % (m, k, n)
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


def generate_db_topi_ops(dot_ops):
    topi_ops = []
    tvm_profiling_log_path = FLAGS.profile_log
    # if os.path.exists(tvm_profiling_log_path):
    #     os.remove(tvm_profiling_log_path)

    for dot_op in dot_ops:
        m = dot_op['arg0_shape'][0]
        k = dot_op['arg0_shape'][1]
        n = dot_op['arg1_shape'][1]
        topi_code = manual_dot_codegen(m, k, n)
        topi_op = {
            'tvm_func_name': get_tvm_topi_func_name(m, k, n),
            'op_type': 'Dot',
            'parameters': dot_op,
            'code': topi_code
        }
        topi_ops.append(topi_op)

    profiling_result = extract_tvm_profiling_from_log(tvm_profiling_log_path)
    for topi_op in topi_ops:
        tvm_func_name = topi_op['tvm_func_name']
        topi_op.update(profiling_result[tvm_func_name])

    return topi_ops


dot_ops = extract_ops_from_log()
topi_ops = generate_db_topi_ops(dot_ops)

with open(FLAGS.output_path, 'w') as fout:
    json.dump(topi_ops, fout)
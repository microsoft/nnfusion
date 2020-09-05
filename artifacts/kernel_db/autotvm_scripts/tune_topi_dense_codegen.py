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
def tvm_matmul_tune_op(batch, in_dim, out_dim):
    """
    autotvm tuning template
    D=A*B
    [batch, in_dim] x [out_dim,in_dim]
    """
    A = tvm.placeholder((batch, in_dim), name='A', dtype="float32")
    B = tvm.placeholder((out_dim, in_dim), name='B', dtype="float32")
    cfg = autotvm.get_config()
    C = topi.cuda.dense.dense_cuda(cfg, A, B)
    s = topi.cuda.dense.schedule_dense(cfg, C)
    cfg.add_flop(batch * in_dim * out_dim * 2)
    return s, [A, B, C]


def tune_dot_codegen(m, k, n):
    logging.getLogger('autotvm').setLevel(logging.DEBUG)
    logging.getLogger('autotvm').addHandler(logging.StreamHandler(sys.stdout))
    task = autotvm.task.create(tvm_matmul_tune_op, args=(m, k, n), target='cuda')

    op_name = "tuned_dot_nt_op_float_m%d_k%d_n%d" % (m, k, n)
    
    log_name = FLAGS.autotvm_log

    dispatch_context = autotvm.apply_history_best(log_name)
    best_config = dispatch_context.query(task.target, task.workload)

    with dispatch_context:
        with tvm.target.create('cuda'):
            s, arg_bufs = tvm_matmul_tune_op(m,k,n)
            func = tvm.build(s, arg_bufs, 'cuda', name=op_name)

    ctx = tvm.context('cuda', 0)

    a_np = np.random.uniform(size=[m,k]).astype("float32")
    w_np = np.random.uniform(size=[n,k]).astype("float32")
    c_np = np.zeros([m,n]).astype("float32")

    a = tvm.nd.array(a_np, ctx)
    w = tvm.nd.array(w_np, ctx)
    c = tvm.nd.array(c_np, ctx)

    kernel_code = func.imported_modules[0].get_source()

    func(a, w, c)

    return kernel_code


def extract_ops_from_log():
    dot_ops = []
    dot_ops.append({'arg0_shape': [1, 256], 'arg1_shape': [256, 256], 'out_shape': [1, 256], 'transpose_A': False, 'transpose_B': True})
    dot_ops.append({'arg0_shape': [1, 256], 'arg1_shape': [256, 512], 'out_shape': [1, 512], 'transpose_A': False, 'transpose_B': True})
    dot_ops.append({'arg0_shape': [1, 512], 'arg1_shape': [512, 1024], 'out_shape': [1, 1024], 'transpose_A': False, 'transpose_B': True})
    dot_ops.append({'arg0_shape': [1, 3008], 'arg1_shape': [3008, 1024], 'out_shape': [1, 1024], 'transpose_A': False, 'transpose_B': True})
    dot_ops.append({'arg0_shape': [4, 256], 'arg1_shape': [256, 256], 'out_shape': [4, 256], 'transpose_A': False, 'transpose_B': True})
    dot_ops.append({'arg0_shape': [4, 256], 'arg1_shape': [256, 512], 'out_shape': [4, 512], 'transpose_A': False, 'transpose_B': True})
    dot_ops.append({'arg0_shape': [4, 512], 'arg1_shape': [512, 1024], 'out_shape': [4, 1024], 'transpose_A': False, 'transpose_B': True})
    dot_ops.append({'arg0_shape': [4, 3008], 'arg1_shape': [3008, 1024], 'out_shape': [4, 1024], 'transpose_A': False, 'transpose_B': True})
    dot_ops.append({'arg0_shape': [16, 256], 'arg1_shape': [256, 256], 'out_shape': [16, 256], 'transpose_A': False, 'transpose_B': True})
    dot_ops.append({'arg0_shape': [16, 256], 'arg1_shape': [256, 512], 'out_shape': [16, 512], 'transpose_A': False, 'transpose_B': True})
    dot_ops.append({'arg0_shape': [16, 512], 'arg1_shape': [512, 1024], 'out_shape': [16, 1024], 'transpose_A': False, 'transpose_B': True})
    dot_ops.append({'arg0_shape': [16, 3008], 'arg1_shape': [3008, 1024], 'out_shape': [16, 1024], 'transpose_A': False, 'transpose_B': True})
    dot_ops.append({'arg0_shape': [1, 4096], 'arg1_shape': [4096, 4096], 'out_shape': [1, 4096], 'transpose_A': False, 'transpose_B': True})
    dot_ops.append({'arg0_shape': [1, 768], 'arg1_shape': [768, 10], 'out_shape': [1, 10], 'transpose_A': False, 'transpose_B': True})
    dot_ops.append({'arg0_shape': [1, 9216], 'arg1_shape': [9216, 4096], 'out_shape': [1, 4096], 'transpose_A': False, 'transpose_B': True})
    return dot_ops


def get_tvm_topi_func_name(m, k, n):
    func_name = "tuned_dot_nt_op_float_m%d_k%d_n%d_kernel0" % (m, k, n)
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

# def get_launch_config(m, k, n):
#     profiling_result = {}


def generate_db_topi_ops(dot_ops):
    topi_ops = []
    # tvm_profiling_log_path = '/home/jxue/vlima/projects/blockfusion-model/kernels/tvm_profile.log'
    # if os.path.exists(tvm_profiling_log_path):
    #     os.remove(tvm_profiling_log_path)
    tvm_profiling_log_path = FLAGS.tvm_profile_log

    for dot_op in dot_ops:
        m = dot_op['arg0_shape'][0]
        k = dot_op['arg0_shape'][1]
        n = dot_op['arg1_shape'][1]
        topi_code = tune_dot_codegen(m, k, n)
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
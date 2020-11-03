"""
matmul autotvm

[batch,in_dim] x [in_dim,out_dim]

search_matmul_config(batch,in_dim,out_dim,num_trials):
    input: batch,in_dim,out_dim,num_trials
        [batch,in_dim] x [in_dim,out_dim]
        num_trials: num of trials, default: 1000
    output: log (json format)
    use autotvm to search configs for the matmul

lookup_matmul_config():
    find a proper matmul config
    note: trade off kernel's performance and grid & block size

launch_matmul_from_config(config):
    input: config (json string)

usage:
    1. use search_matmul_config(batch,in_dim,out_dim,num_trials) to search configs
    2. use lookup_matmul_config() to get a proper config
    3. write the config (in json format) to "matmul_config.json"
    4. use launch_matmul_from_config("matmul_config.json") to print the matmul kernel code
"""


import numpy as np
import tvm
import logging
import sys
from tvm import autotvm
import topi
import json
import os
from topi.util import get_const_tuple
import tensorflow as tf

flags = tf.flags
flags.DEFINE_string("input_path", "", "path of input file")
flags.DEFINE_string("autotvm_log", "../autotvm_logs/all_tuned_tilling_dense_nn.1000.log", "path of autotvm tuning log")
flags.DEFINE_string("tvm_profile_log",
                    "/tmp/tvm_profile.log", "path of tvm profile")
flags.DEFINE_string("output_path", "", "path of output file")

FLAGS = flags.FLAGS


@autotvm.template
def tvm_matmul_tune_op(batch, in_dim, out_dim):
    """
    autotvm tuning template
    D=A*B
    [batch, in_dim] x [in_dim, out_dim]
    """
    A = tvm.placeholder((batch, in_dim), name='A', dtype="float32")
    B = tvm.placeholder((in_dim, out_dim), name='B', dtype="float32")
    k = tvm.reduce_axis((0, in_dim), name='k')
    C = tvm.compute((batch, out_dim), lambda i, j: tvm.sum(
        A[i, k] * B[k, j], axis=k), name='C')

    cfg = autotvm.get_config()
    s = tvm.create_schedule(C.op)
    AA = s.cache_read(A, "shared", [C])
    AL = s.cache_read(AA, "local", [C])
    BB = s.cache_read(B, "shared", [C])
    BL = s.cache_read(BB, "local", [C])
    CC = s.cache_write(C, "local")

    y, x = C.op.axis
    k = CC.op.reduce_axis[0]

    cfg.define_split('tile_k', cfg.axis(k), num_outputs=3)
    ko, kt, ki = cfg['tile_k'].apply(s, CC, k)

    block_x = tvm.thread_axis('blockIdx.x')
    block_y = tvm.thread_axis('blockIdx.y')
    thread_x = tvm.thread_axis('threadIdx.x')
    thread_y = tvm.thread_axis('threadIdx.y')

    cfg.define_split('tile_y', cfg.axis(y), num_outputs=4)
    cfg.define_split('tile_x', cfg.axis(x), num_outputs=4)

    by, tyz, ty, yi = cfg['tile_y'].apply(s, C, y)
    bx, txz, tx, xi = cfg['tile_x'].apply(s, C, x)

    s[C].bind(by, block_y)
    s[C].bind(bx, block_x)
    s[C].bind(tyz, tvm.thread_axis('vthread'))
    s[C].bind(txz, tvm.thread_axis('vthread'))
    s[C].bind(ty, thread_y)
    s[C].bind(tx, thread_x)
    s[C].reorder(by, bx, tyz, txz, ty, tx, yi, xi)

    s[CC].compute_at(s[C], tx)

    yo, xo = CC.op.axis
    s[CC].reorder(ko, kt, yo, xo, ki)
    s[CC].unroll(kt)

    for stage in [AL, BL]:
        s[stage].compute_at(s[CC], kt)
        s[stage].double_buffer()

    for stage in [AA, BB]:
        s[stage].compute_at(s[CC], ko)

        fused = s[stage].fuse(*s[stage].op.axis)
        ty, tx = s[stage].split(fused, nparts=cfg['tile_y'].size[2])
        tx, xi = s[stage].split(tx, nparts=cfg['tile_x'].size[2])
        _, xi = s[stage].split(xi, factor=4)

        s[stage].bind(ty, thread_y)
        s[stage].bind(tx, thread_x)
        s[stage].vectorize(xi)
        s[stage].double_buffer()

    cfg.define_knob('auto_unroll_max_step', [512, 1500])
    s[C].pragma(by, 'auto_unroll_max_step', cfg['auto_unroll_max_step'].val)
    s[C].pragma(by, 'unroll_explicit', False)

    cfg.add_flop(batch * in_dim * out_dim * 2)
    return s, [A, B, C]



def search_matmul_config(batch, in_dim, out_dim, num_trials):

    logging.getLogger('autotvm').setLevel(logging.DEBUG)
    logging.getLogger('autotvm').addHandler(logging.StreamHandler(sys.stdout))
    task = autotvm.task.create(tvm_matmul_tune_op, args=(
        batch, in_dim, out_dim), target='cuda')
    print(task.config_space)
    measure_option = autotvm.measure_option(
        builder=autotvm.LocalBuilder(),
        runner=autotvm.LocalRunner(repeat=3, min_repeat_ms=100, timeout=4)
    )

    op_name = "tuned_dot_op_float_%d_%d_%d" % (batch, in_dim, out_dim)
    log_name = "tuned_kernels/" + op_name + ".log"
    tuner = autotvm.tuner.XGBTuner(task)
    tuner.tune(n_trial=num_trials, measure_option=measure_option,
               callbacks=[autotvm.callback.log_to_file(log_name)])

    dispatch_context = autotvm.apply_history_best(log_name)
    best_config = dispatch_context.query(task.target, task.workload)
    print('\nBest config:')
    print(best_config)

    with dispatch_context:
        with tvm.target.create('cuda'):
            s, arg_bufs = tvm_matmul_tune_op(batch, in_dim, out_dim)
            func = tvm.build(s, arg_bufs, 'cuda', name='matmul')

    ctx = tvm.context('cuda', 0)

    a_np = np.random.uniform(size=(batch, in_dim)).astype("float32")
    b_np = np.random.uniform(size=(in_dim, out_dim)).astype("float32")

    a = tvm.nd.array(a_np, ctx)
    b = tvm.nd.array(b_np, ctx)
    c = tvm.nd.array(np.zeros((batch, out_dim), dtype='float32'), ctx)

    print(func.imported_modules[0].get_source())  # print kernel code

    func(a, b, c)

    num_flops = 2 * batch * in_dim * out_dim
    num_runs = 10
    timer_f = func.time_evaluator(func.entry_name, ctx, number=num_runs)
    t = timer_f(a, b, c).mean
    GFLOPS = num_flops / (t * 1e3) / 1e6
    print("average time cost of %d runs = %g ms, %g GFLOPS." %
          (num_runs, t * 1e3, GFLOPS))


def lookup_matmul_config(batch, in_dim, out_dim, output_log):
    op_name = "tuned_dot_op_float_%d_%d_%d" % (batch, in_dim, out_dim)
    log_name = FLAGS.autotvm_log
    with open(log_name, "r") as fin:
        log_lines = fin.readlines()
    # log_records=tvm.autotvm.record.load_from_file(log_name)
    log_records_all = []
    log_records = []
    for line in log_lines:
        line = line.rstrip('\n')
        # print(line)
        record_json = json.loads(line)
        tm = record_json['r'][0][0]
        if tm > 10000000:  # filter bad configs
            continue
        if record_json['i'][2][0] != batch or record_json['i'][2][1] != in_dim or record_json['i'][2][2] != out_dim:  # filter other configs
            continue
        griddim_x = record_json['i'][5]["e"][2][2][0]
        if griddim_x == -1:
            griddim_x = int(out_dim / record_json['i'][5]["e"][2][2][1] / record_json['i'][5]["e"][2][2][2] / record_json['i'][5]["e"][2][2][3])
        griddim_y = record_json['i'][5]["e"][1][2][0]
        if griddim_y == -1:
            griddim_y = int(batch / record_json['i'][5]["e"][1][2][1] / record_json['i'][5]["e"][1][2][2] / record_json['i'][5]["e"][1][2][3])
        record = {"time": tm,
                  "grid": [griddim_x, griddim_y, 1],
                  "block": [record_json['i'][5]["e"][2][2][2], record_json['i'][5]["e"][1][2][2], 1],
                  "config": line}
        log_records_all.append((tm, record))
        # if record["block"][0] * record["block"][1] * record["block"][2] % 32 != 0:
        #     continue
        # if record["grid"][0] * record["grid"][1] * record["grid"][2] < 16:
        #     continue
        opt = tm * record["grid"][0] * record["grid"][1] * record["grid"][2] * record["block"][0] * record["block"][1] * record["block"][2]
        if record["block"][0] * record["block"][1] * record["block"][2] % 32 != 0:
            opt = tm * record["grid"][0] * record["grid"][1] * record["grid"][2] * (record["block"][0] * record["block"][1] * record["block"][2] / 32 + 1) * 32
        record.update({"opt": opt})
        log_records.append((tm, record))
        # print(log_records[-1])
    log_records_all.sort(key=lambda item: item[0])
    log_records.sort(key=lambda item: item[0])
    print(op_name)
    log_records_fast = log_records[0:100]
    # log_records_fast = log_records
    log_records = []
    for i in range(len(log_records_fast)):
        log_records.append((log_records_fast[i][1]["opt"], log_records_fast[i][1]))
    log_records.sort(key=lambda item: item[0])
    print("fastest kernel:", log_records_all[0][1]["time"], "grid:", log_records_all[0][1]["grid"], "block:", log_records_all[0][1]["block"])
    # print(log_records_fast[0][1]["config"])
    print("efficient kernel:",log_records[0][1]["time"], "grid:", log_records[0][1]["grid"], "block:", log_records[0][1]["block"])
    with open(output_log, 'a') as fout:
        fout.write(log_records[0][1]["config"] + "\n")


def launch_matmul_from_config(config_json_path):
    with open(config_json_path, "r") as fin:
        config = json.load(fin)

    batch = config["i"][2][0]
    in_dim = config["i"][2][1]
    out_dim = config["i"][2][2]

    # print(batch, in_dim, out_dim)

    task = autotvm.task.create(
        tvm_matmul_tune_op, args=(batch, in_dim, out_dim), target='cuda')
    # dispatch_context = autotvm.task.ApplyConfig(config)
    dispatch_context = autotvm.apply_history_best(config_json_path)
    best_config = dispatch_context.query(task.target, task.workload)
    print("Using pretuned config:")
    print(best_config)

    with dispatch_context:
        with tvm.target.create('cuda'):
            s, arg_bufs = tvm_matmul_tune_op(batch, in_dim, out_dim)
            func = tvm.build(s, arg_bufs, 'cuda', name='matmul')

    ctx = tvm.context('cuda', 0)

    a_np = np.random.uniform(size=(batch, in_dim)).astype("float32")
    b_np = np.random.uniform(size=(in_dim, out_dim)).astype("float32")

    a = tvm.nd.array(a_np, ctx)
    b = tvm.nd.array(b_np, ctx)
    c = tvm.nd.array(np.zeros((batch, out_dim), dtype='float32'), ctx)

    print(func.imported_modules[0].get_source())  # print kernel code

    func(a, b, c)

    num_flops = 2 * batch * in_dim * out_dim
    num_runs = 10
    timer_f = func.time_evaluator(func.entry_name, ctx, number=num_runs)
    t = timer_f(a, b, c).mean
    GFLOPS = num_flops / (t * 1e3) / 1e6
    print("average time cost of %d runs = %g ms, %g GFLOPS." %
          (num_runs, t * 1e3, GFLOPS))




output_log_file = "matmul_nn_autotvm_select_result.log"

if os.path.exists(output_log_file):
    os.remove(output_log_file)

lookup_matmul_config(4, 256, 256, output_log_file)
lookup_matmul_config(16, 256, 256, output_log_file)




def tune_dot_codegen(m, k, n, log_path):
    logging.getLogger('autotvm').setLevel(logging.DEBUG)
    logging.getLogger('autotvm').addHandler(logging.StreamHandler(sys.stdout))
    task = autotvm.task.create(tvm_matmul_tune_op, args=(m, k, n), target='cuda')

    op_name = "tuned_dot_nn_op_float_m%d_k%d_n%d" % (m, k, n)
    
    # log_name = "tuned_dot_op_float_%d_%d_%d" % (m, k, n)
    # log_name = "tuned_kernels/" + log_name + ".log"
    log_name = log_path

    dispatch_context = autotvm.apply_history_best(log_name)
    best_config = dispatch_context.query(task.target, task.workload)

    with dispatch_context:
        with tvm.target.create('cuda'):
            s, arg_bufs = tvm_matmul_tune_op(m,k,n)
            func = tvm.build(s, arg_bufs, 'cuda', name=op_name)

    ctx = tvm.context('cuda', 0)

    a_np = np.random.uniform(size=[m,k]).astype("float32")
    w_np = np.random.uniform(size=[k,n]).astype("float32")
    c_np = np.zeros([m,n]).astype("float32")

    a = tvm.nd.array(a_np, ctx)
    w = tvm.nd.array(w_np, ctx)
    c = tvm.nd.array(c_np, ctx)

    kernel_code = func.imported_modules[0].get_source()

    func(a, w, c)

    return kernel_code



def extract_ops_from_log():
    dot_ops = []
    dot_ops.append({'arg0_shape': [4, 256], 'arg1_shape': [256, 256], 'out_shape': [4, 256], 'transpose_A': False, 'transpose_B': False})
    dot_ops.append({'arg0_shape': [16, 256], 'arg1_shape': [256, 256], 'out_shape': [16, 256], 'transpose_A': False, 'transpose_B': False})
    return dot_ops


def get_tvm_topi_func_name(m, k, n):
    func_name = "tuned_dot_nn_op_float_m%d_k%d_n%d_kernel0" % (m, k, n)
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


def generate_db_topi_ops(dot_ops, log_path):
    topi_ops = []
    tvm_profiling_log_path = FLAGS.tvm_profile_log
    if os.path.exists(tvm_profiling_log_path):
        os.remove(tvm_profiling_log_path)

    for dot_op in dot_ops:
        m = dot_op['arg0_shape'][0]
        k = dot_op['arg0_shape'][1]
        n = dot_op['arg1_shape'][1]
        topi_code = tune_dot_codegen(m, k, n, log_path)
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
topi_ops = generate_db_topi_ops(dot_ops, output_log_file)

with open(FLAGS.output_path, 'w') as fout:
    json.dump(topi_ops, fout)

os.remove(output_log_file)
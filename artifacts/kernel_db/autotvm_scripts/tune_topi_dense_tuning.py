"""
matmul autotvm

[batch,in_dim] x [out_dim,in_dim]

search_matmul_config(batch,in_dim,out_dim,num_trials):
    input: batch,in_dim,out_dim,num_trials
        [batch,in_dim] x [out_dim,in_dim]
        num_trials: num of trials, default: 1000
    output: log (json format)
    use autotvm to search configs for the matmul
"""


import numpy as np
import tvm
import logging
import sys
from tvm import autotvm
import topi
import json
from topi.util import get_const_tuple


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
    log_name = "tuned_topi_dense_kernels/" + op_name + ".log"

    tuner = autotvm.tuner.XGBTuner(task)
    tsk_trial = min(num_trials, len(task.config_space))
    tuner.tune(n_trial=tsk_trial, measure_option=measure_option,
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
    b_np = np.random.uniform(size=(out_dim, in_dim)).astype("float32")

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


# lstm, deepspeech2, seq2seq
search_matmul_config(1, 256, 256, 1000)  # m, k, n, num_trials
search_matmul_config(1, 256, 512, 1000)  # m, k, n, num_trials
search_matmul_config(1, 512, 1024, 1000)  # m, k, n, num_trials
search_matmul_config(1, 3008, 1024, 1000)  # m, k, n, num_trials
search_matmul_config(4, 256, 256, 1000)  # m, k, n, num_trials
search_matmul_config(4, 256, 512, 1000)  # m, k, n, num_trials
search_matmul_config(4, 512, 1024, 1000)  # m, k, n, num_trials
search_matmul_config(4, 3008, 1024, 1000)  # m, k, n, num_trials
search_matmul_config(16, 256, 256, 1000)  # m, k, n, num_trials
search_matmul_config(16, 256, 512, 1000)  # m, k, n, num_trials
search_matmul_config(16, 512, 1024, 1000)  # m, k, n, num_trials
search_matmul_config(16, 3008, 1024, 1000)  # m, k, n, num_trials

# deepspeech2
search_matmul_config(75, 256, 29, 1000)

# alexnet
search_matmul_config(1, 4096, 1000, 1000)
search_matmul_config(1, 4096, 4096, 1000)
search_matmul_config(1, 9216, 4096, 1000)

# resnext
search_matmul_config(1, 256, 10, 1000)
search_matmul_config(4, 256, 10, 1000)
search_matmul_config(16, 256, 10, 1000)


# nasnet_cifar
search_matmul_config(1, 768, 10, 1000)

# resnext-imagenet
search_matmul_config(1, 2048, 1000, 1000)

# nasnet-imagenet
search_matmul_config(1, 1056, 1000, 1000)
import numpy as np
import tvm
import logging
import sys
from tvm import autotvm
from tvm import te
import json
import re

# log path
LOG_PATH = 'matmul_autotvm.log'


@autotvm.template("matmul")
def tvm_matmul_tune_op(batch, in_dim, out_dim, transA, transB):
    """
    autotvm tuning template
    D=A*B
    [batch, in_dim] x [in_dim, out_dim]
    """
    if not transA and not transB:
        A = te.placeholder((batch, in_dim), name='A', dtype="float32")
        B = te.placeholder((in_dim, out_dim), name='B', dtype="float32")
        k = te.reduce_axis((0, in_dim), name='k')
        C = te.compute((batch, out_dim), lambda i, j: te.sum(
            A[i, k] * B[k, j], axis=k), name='C')
    elif transA and not transB:
        A = te.placeholder((in_dim, batch), name='A', dtype="float32")
        B = te.placeholder((in_dim, out_dim), name='B', dtype="float32")
        k = te.reduce_axis((0, in_dim), name='k')
        C = te.compute((batch, out_dim), lambda i, j: te.sum(
            A[k, i] * B[k, j], axis=k), name='C')
    elif not transA and transB:
        A = te.placeholder((batch, in_dim), name='A', dtype="float32")
        B = te.placeholder((out_dim, in_dim), name='B', dtype="float32")
        k = te.reduce_axis((0, in_dim), name='k')
        C = te.compute((batch, out_dim), lambda i, j: te.sum(
            A[i, k] * B[j, k], axis=k), name='C')
    cfg = autotvm.get_config()
    s = te.create_schedule(C.op)
    AA = s.cache_read(A, "shared", [C])
    AL = s.cache_read(AA, "local", [C])
    BB = s.cache_read(B, "shared", [C])
    BL = s.cache_read(BB, "local", [C])
    CC = s.cache_write(C, "local")

    y, x = C.op.axis
    k = CC.op.reduce_axis[0]

    cfg.define_split('tile_k', cfg.axis(k), num_outputs=3)
    ko, kt, ki = cfg['tile_k'].apply(s, CC, k)

    block_x = te.thread_axis('blockIdx.x')
    block_y = te.thread_axis('blockIdx.y')
    thread_x = te.thread_axis('threadIdx.x')
    thread_y = te.thread_axis('threadIdx.y')

    cfg.define_split('tile_y', cfg.axis(y), num_outputs=4)
    cfg.define_split('tile_x', cfg.axis(x), num_outputs=4, policy='power2')

    by, tyz, ty, yi = cfg['tile_y'].apply(s, C, y)
    bx, txz, tx, xi = cfg['tile_x'].apply(s, C, x)

    s[C].bind(by, block_y)
    s[C].bind(bx, block_x)
    s[C].bind(tyz, te.thread_axis('vthread'))
    s[C].bind(txz, te.thread_axis('vthread'))
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


def search_matmul_config(batch, in_dim, out_dim, transA, transB, num_trials):
    logging.getLogger('autotvm').setLevel(logging.DEBUG)
    logging.getLogger('autotvm').addHandler(logging.StreamHandler(sys.stdout))
    task = autotvm.task.create("matmul", args=(
        batch, in_dim, out_dim, transA, transB), target='cuda')
    print(task.config_space)
    measure_option = autotvm.measure_option(
        builder=autotvm.LocalBuilder(),
        runner=autotvm.LocalRunner(repeat=3, min_repeat_ms=100, timeout=4)
    )

    op_name = f"tuned_dot_op_float_{batch}_{in_dim}_{out_dim}_{transA}_{transB}"
    print(op_name)
    log_name = "autotvm_kernels/" + op_name + ".log"
    tuner = autotvm.tuner.XGBTuner(task)
    tuner.tune(n_trial=num_trials, measure_option=measure_option,
               callbacks=[autotvm.callback.log_to_file(log_name)])

    dispatch_context = autotvm.apply_history_best(log_name)
    best_config = dispatch_context.query(task.target, task.workload)
    print('\nBest config:')
    print(best_config)

    with dispatch_context:
        with tvm.target.create('cuda'):
            s, arg_bufs = tvm_matmul_tune_op(batch, in_dim, out_dim, transA, transB)
            func = tvm.build(s, arg_bufs, 'cuda', name='matmul')

    ctx = tvm.context('cuda', 0)

    if transA:
        a_np = np.random.uniform(size=(in_dim, batch)).astype("float32")
    else:
        a_np = np.random.uniform(size=(batch, in_dim)).astype("float32")

    if transB:
        b_np = np.random.uniform(size=(out_dim, in_dim)).astype("float32")
    else:
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


def codegen_and_inject(batch, in_dim, out_dim, transA, transB):
    logging.getLogger('autotvm').setLevel(logging.DEBUG)
    logging.getLogger('autotvm').addHandler(logging.StreamHandler(sys.stdout))
    task = autotvm.task.create("matmul", args=(
        batch, in_dim, out_dim, transA, transB), target='cuda')
    print(task.config_space)

    op_name = f"tuned_dot_op_float_{batch}_{in_dim}_{out_dim}_{transA}_{transB}"
    print(op_name)
    log_name = "autotvm_kernels/" + op_name + ".log"

    dispatch_context = autotvm.apply_history_best(log_name)
    pattern = re.compile(r'attr \[IterVar\(((?:blockIdx|threadIdx)).([xyz]): int32, \(nullptr\), "ThreadIndex", "((?:blockIdx|threadIdx)).([xyz])"\)\] "thread_extent" = (\d+)')
    with dispatch_context:
        with tvm.target.create('cuda'):
            s, arg_bufs = tvm_matmul_tune_op(batch, in_dim, out_dim, transA, transB)
            func = tvm.build(s, arg_bufs, 'cuda', name="default_function")
            code = func.imported_modules[0].get_source()
            lowered_text = str(tvm.lower(s, arg_bufs, simple_mode=True))
            launch_config = {
                'blockIdx.x': 1,
                'blockIdx.y': 1,
                'blockIdx.z': 1,
                'threadIdx.x': 1,
                'threadIdx.y': 1,
                'threadIdx.z': 1,
            }
            for st in lowered_text.splitlines():
                attr = pattern.search(st)
                if attr is not None:
                    assert(attr.group(1) == attr.group(3)) # thread/block
                    assert(attr.group(2) == attr.group(4)) # xyz
                    launch_config[f"{attr.group(1)}.{attr.group(2)}"] = int(attr.group(5))
    
    with open("autotvm_kernels/" + op_name + ".cu", 'w') as f:
        f.write(r"// %%%")
        f.write('\n')
        f.write(code)
        f.write(r"// %%%")
        f.write('\n')
        f.write(r"// +++")
        f.write('\n')
        f.write(f"dim3 grid({launch_config['blockIdx.x']}, {launch_config['blockIdx.y']}, {launch_config['blockIdx.z']});\n")
        f.write(f"dim3 block({launch_config['threadIdx.x']}, {launch_config['threadIdx.y']}, {launch_config['threadIdx.z']});\n")
        f.write(r"// +++")
        f.write('\n')

    best_grid_size = tuple((launch_config['blockIdx.x'], launch_config['blockIdx.y'], launch_config['blockIdx.z']))
    best_block_size = tuple((launch_config['threadIdx.x'], launch_config['threadIdx.y'], launch_config['threadIdx.z']))

    from db.save_to_db import save_to_db
    if transA:
        A1, A2 = in_dim, batch
    else:
        A1, A2 = batch, in_dim
    if transB:
        B1, B2 = out_dim, in_dim
    else:
        B1, B2 = in_dim, out_dim
    C1, C2 = batch, out_dim
    save_to_db(f"Dot[{A1},{A2};{B1},{B2};{C1},{C2}floatfloatfloat{int(transA)}{int(transB)}]", code, best_grid_size, best_block_size)


if __name__ == '__main__':
    # nt = 10000
    # search_matmul_config(64, 256, 3797, False, True, nt)
    codegen_and_inject(64, 256, 3797, False, True)

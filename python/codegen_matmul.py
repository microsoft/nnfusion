import numpy as np
import tvm
from tvm import te

from d2ltvm import matmul, bench_workload
import memopt

# Defined in file: ./chapter_getting_started/vector_add.md
def get_data(n, m, k, constructor=None):
    """Return random a, b and empty c with the same shape.
    """
    np.random.seed(0)
    a = np.random.normal(size=(n, k)).astype(np.float32)
    b = np.random.normal(size=(k, m)).astype(np.float32)
    c = np.empty((n, m), dtype='float32')
    if constructor:
        a, b, c = [constructor(x) for x in (a, b, c)]
    return a, b, c

# Defined in file: ./chapter_cpu_schedules/matmul.md
def bench_matmul_tvm(func, sizes, target):
    def workload(nrepeats):
        timer = mod.time_evaluator(mod.entry_name, dev=ctx, number=nrepeats)
        return timer(a, b, c).mean * nrepeats
    times = []
    for size in sizes:
        n, m, k = size
        s, (A, B, C) = func(n, m, k)
        mod = tvm.build(s, [A, B, C], target)
        ctx = tvm.device(target, 3)
        a, b, c = get_data(n, m, k, lambda x: tvm.nd.array(x, device=ctx))
        times.append(bench_workload(workload))
    return np.array(times)

tx, ty, tk = 8, 4, 32  # tile sizes for one CUDA thread

def vector_add(n):
    """TVM expression for vector add"""
    A = te.placeholder((n,), name='A')
    B = te.placeholder((n,), name='B')
    C = te.compute(A.shape, lambda i: A[i] + B[i], name='C')
    s = te.create_schedule(C.op)
    x = s[C].op.axis[0]
    xb, xo, xi = split(s[C], x, (16, 32))
    s[C].reorder(xb, xi, xo)
    bind_thread(s[C], (xb, xi), ("blockIdx.x", "threadIdx.x"))
    return s, (A, B, C)

def tile_matmul(n, m, k):
    A, B, C = matmul(n, m, k)
    s = te.create_schedule(C.op)
    # Create caches
    A_shared = s.cache_read(A, "shared", [C])
    A_local  = s.cache_read(A_shared, "local", [C])
    B_shared = s.cache_read(B, "shared", [C])
    B_local  = s.cache_read(B_shared, "local", [C])
    C_local = s.cache_write(C, "local")
    # Split each axis into block axis, thread axis, and inner axis
    x, y = s[C].op.axis
    block_size = [32, 16]
    xb, xo, xi = split(s[C], x, (block_size[0], 4))
    yb, yo, yi = split(s[C], y, (block_size[1], 4))
    s[C].reorder(xb, yb, xo, yo, xi, yi)
    # Note that we bind yb to blockIdx.x instead of blockIdx.y
    bind_thread(s[C], (xb, yb, yo, xo),
                ("blockIdx.x", "blockIdx.y", "threadIdx.x", "threadIdx.y"))
    # Schedule C_local
    s[C_local].compute_at(s[C], yo)
    yi, xi = s[C_local].op.axis
    k, = s[C_local].op.reduce_axis
    ko, ki = s[C_local].split(k, 8)
    s[C_local].reorder(ko, ki, yi, xi)
    # Optimize read caches of A and B with cooperative fetching
    def optimize_read_cache(shared, local):
        s[shared].compute_at(s[C_local], ko)
        s[local].compute_at(s[C_local], ki)
        args = s[shared].op.axis
        fused = s[shared].fuse(*args)
        # Note that we must split into block_size parts to reuse
        # the previous axis threads
        yo, fused = s[shared].split(fused, nparts=block_size[0])
        xo, _ = s[shared].split(fused, nparts=block_size[1])
        # s[shared].reorder(yo, xo, yi, xi)
        bind_thread(s[shared], (yo, xo), ("threadIdx.y", "threadIdx.x"))
    # def at_root(shared, local):
    #     s[shared].compute_at(s[C], xo)
    #     s[local].compute_at(s[C_local], ki)
    #     args = s[shared].op.axis
    #     fused = s[shared].fuse(*args)
    #     fo, fi = s[shared].split(fused, factor=16)
    #     fy, fx = s[shared].split(fo, factor=32)
    #     s[shared].bind(fx, te.thread_axis("threadIdx.x"))
    #     s[shared].bind(fy, te.thread_axis("threadIdx.y"))
    optimize_read_cache(A_shared, A_local)
    optimize_read_cache(B_shared, B_local)
    return s, (A, B, C)

# Save into the d2ltvm package.
def split(stage, axis, factors):
    """Split an axis by a list of factors in a reverse order
    """
    axes = []
    for f in reversed(factors):
        axis, x = stage.split(axis, f)
        axes.append(x)
    return list(reversed(axes+[axis]))

# Save into the d2ltvm package.
def bind_thread(stage, axes, tags):
    """Bind a list of axes to thread axes
    """
    for axis, tag in zip(axes, tags):
        stage.bind(axis, te.thread_axis(tag))

target = tvm.target.cuda(arch="sm_61")
n, m, k = 4096, 128, 128
sch, (X, K, Y) = tile_matmul(n, m, k)
# sch, (X, K, Y) = vector_add(3333)
passes = [
    # (0, memopt.debug_pass),
    # (0, memopt.modify_output_pass),
    # (2, memopt.get_kernel_info_pass),
]
with tvm.transform.PassContext(config={"tir.add_lower_pass": passes}), \
    memopt.Scope(sch) as scope:
    scope.shared_mem_outputs = ["C"]
    mod = tvm.build(sch, [X, K, Y], target=target)

# print(tvm.lower(sch, [X, K, Y], simple_mode=True))
# data, weight, out = get_conv_data(oc, ic, n, k, p, s, tvm.nd.array)
# mod(data, weight, out)

kernel_code = mod.imported_modules[0].get_source()
print(kernel_code)
tms = bench_matmul_tvm(tile_matmul, [(n, m, k)], 'cuda')
print("Result", tms)

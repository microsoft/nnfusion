import numpy as np
import tvm
from tvm import te

from d2ltvm import split_axis, conv_out_size, padding, get_conv_data, bench_conv_tvm

def conv(oc, ic, nh, nw, kh, kw, ph=0, pw=0, sh=1, sw=1):
    """Convolution
    oc, ic : output and input channels
    nh, nw : input width and height
    kh, kw : kernel width and height
    ph, pw : height and width padding sizes, default 0
    sh, sw : height and width strides, default 1
    """
    # reduction axes
    ric = te.reduce_axis((0, ic), name='ric')
    rkh = te.reduce_axis((0, kh), name='rkh')
    rkw = te.reduce_axis((0, kw), name='rkw')
    # output height and weights
    oh = conv_out_size(nh, kh, ph, sh)
    ow = conv_out_size(nw, kw, pw, sw)
    # pad X and then compute Y
    X = te.placeholder((ic, nh, nw), name='X')
    K = te.placeholder((oc, ic, kh, kw), name='K')
    Y = te.compute(
        (oc, oh, ow),
        lambda c, i, j: te.sum(
            X[ric, i*sh, j*sw] * K[c, ric, 0, 0],
            axis=[ric]), name='Y')
    return X, K, Y


def default_sch(oc, ic, n, k, p, s):
    X, K, Y = conv(oc, ic, n, n, k, k, p, p, s, s)
    sch = te.create_schedule(Y.op)
    c, y, x = sch[Y].op.axis
    sch[Y].bind(c, te.thread_axis("blockIdx.x"))
    sch[Y].bind(y, te.thread_axis("threadIdx.x"))
    print(tvm.lower(sch, [X, K, Y], simple_mode=True))
    return sch, (X, K, Y)

def tiling(oc, ic, n, k, p, s):
    tile_c = [8, 8]
    tile_h = [4, 2]
    tile_w = [4, 2]
    tile_rc = [1, 1]
    X, K, Y = conv(oc, ic, n, n, k, k, p, p, s, s)
    sch = te.create_schedule(Y.op)
    YL = sch.cache_write(Y, 'local')
    # create cache stage
    XX = sch.cache_read(X, 'shared', [YL])
    KK = sch.cache_read(K, 'shared', [YL])
    XL = sch.cache_read(XX, 'local', [YL])
    KL = sch.cache_read(KK, 'local', [YL])
    c, h, w = sch[Y].op.axis
    bc, tc, ic = split_axis(tile_c, sch, Y, c)
    bh, th, ih = split_axis(tile_h, sch, Y, h)
    bw, tw, iw = split_axis(tile_w, sch, Y, w)
    sch[Y].bind(bc, te.thread_axis("blockIdx.z"))
    sch[Y].bind(bh, te.thread_axis("blockIdx.y"))
    sch[Y].bind(bw, te.thread_axis("blockIdx.x"))
    sch[Y].bind(tc, te.thread_axis("threadIdx.z"))
    sch[Y].bind(th, te.thread_axis("threadIdx.y"))
    sch[Y].bind(tw, te.thread_axis("threadIdx.x"))
    sch[Y].reorder(bc, bh, bw, tc, th, tw, ic, ih, iw)
    sch[YL].compute_at(sch[Y], tw)
    # tile reduction axes
    c, h, w = sch[YL].op.axis
    rc = sch[YL].op.reduce_axis[0]
    rco, rcm, rci = split_axis(tile_rc, sch, YL, rc)
    sch[YL].reorder(rco, rcm, rci, c, h, w)
    sch[XX].compute_at(sch[YL], rco)
    sch[KK].compute_at(sch[YL], rco)
    sch[XL].compute_at(sch[YL], rcm)
    sch[KL].compute_at(sch[YL], rcm)
    # cooperative fetching
    for load in [XX, KK]:
        args = sch[load].op.axis
        fused = sch[load].fuse(*args)
        # align thread layout
        tz, fused = sch[load].split(fused, nparts=tile_c[0])
        ty, fused = sch[load].split(fused, nparts=tile_h[0])
        tx, _ = sch[load].split(fused, nparts=tile_w[0])
        sch[load].bind(tz, te.thread_axis("threadIdx.z"))
        sch[load].bind(ty, te.thread_axis("threadIdx.y"))
        sch[load].bind(tx, te.thread_axis("threadIdx.x"))
    return sch, (X, K, Y)

def vthread(oc, ic, n, k, p, s):
    tile_c = [1, 4, 8]
    tile_h = [1, 2, 2]
    tile_w = [2, 16, 2] # making 2 virtual thread along the ow dimension
    tile_rc = [1, 1]
    tile_rh = [1, 3] # making the data access in columns
    tile_rw = [1, 1]
    X, K, Y, PaddedX = conv(oc, ic, n, n, k, k, p, p, s, s)
    sch = te.create_schedule(Y.op)
    sch[PaddedX].compute_inline()
    YL = sch.cache_write(Y, 'local')
    # create cache stage
    XX = sch.cache_read(PaddedX, 'shared', [YL])
    KK = sch.cache_read(K, 'shared', [YL])
    XL = sch.cache_read(XX, 'local', [YL])
    KL = sch.cache_read(KK, 'local', [YL])
    c, h, w = sch[Y].op.axis
    bc, vc, tc, ic = split_axis(tile_c, sch, Y, c)
    bh, vh, th, ih = split_axis(tile_h, sch, Y, h)
    bw, vw, tw, iw = split_axis(tile_w, sch, Y, w)
    sch[Y].bind(bc, te.thread_axis("blockIdx.z"))
    sch[Y].bind(bh, te.thread_axis("blockIdx.y"))
    sch[Y].bind(bw, te.thread_axis("blockIdx.x"))
    sch[Y].bind(vc, te.thread_axis("vthread"))
    sch[Y].bind(vh, te.thread_axis("vthread"))
    sch[Y].bind(vw, te.thread_axis("vthread"))
    sch[Y].bind(tc, te.thread_axis("threadIdx.z"))
    sch[Y].bind(th, te.thread_axis("threadIdx.y"))
    sch[Y].bind(tw, te.thread_axis("threadIdx.x"))
    sch[Y].reorder(bc, bh, bw, vc, vh, vw, tc, th, tw, ic, ih, iw)
    sch[YL].compute_at(sch[Y], tw)
    # tile reduction axes
    c, h, w = sch[YL].op.axis
    rc, rh, rw = sch[YL].op.reduce_axis
    rco, rcm, rci = split_axis(tile_rc, sch, YL, rc)
    rho, rhm, rhi = split_axis(tile_rh, sch, YL, rh)
    rwo, rwm, rwi = split_axis(tile_rw, sch, YL, rw)
    sch[YL].reorder(rco, rho, rwo, rcm, rhm, rwm, rci, rhi, rwi, c, h, w)
    sch[XX].compute_at(sch[YL], rwo)
    sch[KK].compute_at(sch[YL], rwo)
    sch[XL].compute_at(sch[YL], rwm)
    sch[KL].compute_at(sch[YL], rwm)
    # cooperative fetching
    for load in [XX, KK]:
        args = sch[load].op.axis
        fused = sch[load].fuse(*args)
        # align thread layout
        tz, fused = sch[load].split(fused, nparts=tile_c[1])
        ty, fused = sch[load].split(fused, nparts=tile_h[1])
        tx, _ = sch[load].split(fused, nparts=tile_w[1])
        sch[load].bind(tz, te.thread_axis("threadIdx.z"))
        sch[load].bind(ty, te.thread_axis("threadIdx.y"))
        sch[load].bind(tx, te.thread_axis("threadIdx.x"))

    return sch, (X, K, Y)

target = tvm.target.cuda(arch="sm_61")
oc, ic, n, k, p, s = 64, 64, 64, 1, 0, 1
sch, (X, K, Y) = tiling(oc, ic, n, k, p, s)
mod = tvm.build(sch, [X, K, Y], target=target)

kernel_code = mod.imported_modules[0].get_source()
print(kernel_code)
tvm_gflops = bench_conv_tvm(tiling, [(64, 64, 1)], 'cuda')
print("Result", tvm_gflops)

import numpy as np
import sys
import tvm
from tvm import te, topi, testing
from tvm.topi.testing import conv2d_nchw_python
import tvm.testing
from tvm import te
from codegen import CodeGenerator
import sys
from tvm.topi import nn
from tvm.contrib import nvcc

@tvm.register_func("tvm_callback_cuda_compile", override=True)
def tvm_callback_cuda_compile(code):
    ptx = nvcc.compile_cuda(code, target="ptx", arch="compute_70")
    return ptx

tile_n = [1, 2, 1]
tile_f = [4, 4, 1]
tile_h = [8, 1, 1]
tile_w = [1, 16, 1]
tile_rc = [1, 1]
tile_rx = [1, 1]
tile_ry = [1, 1]

N, CI, H, W, CO, KH, KW, strides, padding = 64, 3, 230, 230, 64, 7, 7, (2, 2), (0, 0)

if len(sys.argv) == 11:
    N = int(sys.argv[1])
    CI = int(sys.argv[2])
    H = int(sys.argv[3])
    W = int(sys.argv[4])
    CO = int(sys.argv[5])
    KH = int(sys.argv[6])
    KW = int(sys.argv[7])
    s = (int(sys.argv[8]), int(sys.argv[8]))
    d = (int(sys.argv[9]), int(sys.argv[9]))
    p = (int(sys.argv[10]), int(sys.argv[10]))

print(N, CI, H, W, CO, KH, KW, strides, padding)

def split_axis(factors, sch, op, axis):
    ret = []
    for i in range(0, len(factors)):
        ax0, ax1 = sch[op].split(axis, factor=int(np.prod(factors[i:])))
        ret.append(ax0)
        axis = ax1
    return ret + [axis]


def conv():
    data = te.placeholder((N, CI, H, W), name="data")
    kernel = te.placeholder((CO, CI, KH, KW), name="kernel")
    conv = nn.conv2d_nchw(data, kernel, strides, padding, 1)
    s = te.create_schedule(conv.op)

    tile_dict = {'rc': [1], 'nn': [2, 2, 1], 'ff': [4, 1, 1], 'yy': [1, 4, 1], 'xx': [1, 4, 1], 'ry': [1, 1], 'rx': [1, 1]}
    generator = CodeGenerator()
    generator.rewrite_schedule(s, tile_dict, True, True)
    func = tvm.build(s, [data, kernel, conv], "cuda")

    # s[temp].compute_inline()
    # n, f, y, x = s[conv].op.axis
    # rc, ry, rx = s[conv].op.reduce_axis

    # output = conv

    # # create cache stage
    # OL = s.cache_write(conv, "local")
    # AA = s.cache_read(temp, "shared", [OL])
    # WW = s.cache_read(kernel, "shared", [OL])
    # AL = s.cache_read(AA, "local", [OL])
    # WL = s.cache_read(WW, "local", [OL])

    # # tile and bind spatial axes
    # n, f, y, x = s[output].op.axis
    # bn, vn, tn, ni = split_axis(tile_n, s, output, n)
    # bf, vf, tf, fi = split_axis(tile_f, s, output, f)
    # by, vy, ty, yi = split_axis(tile_h, s, output, y)
    # bx, vx, tx, xi = split_axis(tile_w, s, output, x)
    # s[output].reorder(bn, bf, by, bx, vn, vf, vy, vx, tn, tf, ty, tx, ni, fi, yi, xi)

    # blck = s[output].fuse(bn, bf, by, bx)
    # vthd = s[output].fuse(vn, vf, vy, vx)
    # thrd = s[output].fuse(tn, tf, ty, tx)


    # s[output].bind(blck, te.thread_axis("blockIdx.x"))
    # s[output].bind(vthd, te.thread_axis("vthread"))
    # s[output].bind(thrd, te.thread_axis("threadIdx.x"))
    # s[output].reorder(blck, vthd, thrd, ni, fi, yi, xi)
    # # s[output].unroll(ni)
    # # s[output].unroll(fi)
    # # s[output].unroll(yi)
    # # s[output].unroll(xi)

    # # s[OL].compute_at(s[output], tx)
    # s[OL].compute_at(s[output], thrd)

    # # tile reduction axes
    # n, f, y, x = s[OL].op.axis
    # # xo, xi = s[OL].split(x, 2)
    # rc, ry, rx = s[OL].op.reduce_axis
    # # rco, rcm, rci = split_axis(tile_rc, s, OL, rc)
    # # ryo, rym, ryi = split_axis(tile_ry, s, OL, ry)
    # # rxo, rxm, rxi = split_axis(tile_rx, s, OL, rx)

    # # s[OL].reorder(rco, rcm, rci, ryo, rym, ryi, rxo, rxm, rxi, n, f, y, x)
    # s[OL].reorder(rc, ry, rx, n, f, y, x)
    # fused = s[OL].fuse(n, f, y, x)
    # s[OL].unroll(fused)
    # s[AA].compute_at(s[OL], rc)
    # s[WW].compute_at(s[OL], rc)
    # s[AL].compute_at(s[OL], rx)
    # s[WL].compute_at(s[OL], rx)

    # # cooperative fetching
    # for load in [AA, WW]:
    #     n, f, y, x = s[load].op.axis
    #     fused = s[load].fuse(n, f, y, x)
    #     oo, ii = s[load].split(fused, factor=tile_n[1]*tile_f[1]*tile_h[1]*tile_w[1])
    #     s[load].reorder(oo, ii)
    #     s[load].unroll(oo)
    #     s[load].bind(ii, te.thread_axis("threadIdx.x"))

    # func = tvm.build(s, [data, kernel, conv], "cuda")
    with open('conv.cuh', 'w') as ouf:
        ouf.write('#ifndef KERNELH\n#define KERNELH\n')
        ouf.write(func.imported_modules[0].get_source())
        ouf.write('#endif\n')

    ctx = tvm.gpu(0)
    val_a = 0.1
    val_b = 0.2
    a_np = np.full((N, CI, H, W), val_a, 'float32')
    w_np = np.full((CO, CI, KH, KW), val_b, 'float32')

    a = tvm.nd.array(a_np, ctx)
    w = tvm.nd.array(w_np, ctx)
    OH = (H - KH + padding[0] * 2) // strides[0] + 1
    OW = (W - KW + padding[1] * 2) // strides[1] + 1
    b = tvm.nd.array(np.zeros((N, CO, OH, OW), dtype=conv.dtype), ctx)
    func(a, w, b)

    # Check correctness

    # result = b.asnumpy()
    # prod = val_a * val_b * CI
    # for ni in range(10):
    #     for fi in range(CO):
    #         for hi in range(OH):
    #             for wi in range(OW):
    #                 if abs(result[ni][fi][hi][wi] - 2.94) > 1e-3:
    #                     print('Error at {}, {}, {}, {}: {} {}'.format(ni, fi, hi, wi, result[ni][fi][hi][wi], 2.94))

    evaluator = func.time_evaluator(func.entry_name, ctx, number=1)
    print("Convolution: %f ms" % (evaluator(a, w, b).mean * 1e3))

# for vf in [1, 2, 4]:
#     tile_f[0] = vf
#     tile_f[2] = 4 // vf
#     for vy in [1, 2]:
#         tile_h[0] = vy
#         tile_h[2] = 2 // vy
#         for vx in [1, 2, 4]:
#             tile_w[0] = vx
#             tile_w[2] = 4 // vx
#             print(tile_f, tile_h, tile_w)
conv()

# generate kernel entry
# out_height = (H - KH + padding[0] * 2) // strides[0] + 1
# out_width = (W - KW + padding[1] * 2) // strides[1] + 1

# tvm_func_name = "tuned_fused_convolution_op_float_i{0}_{1}_{2}_{3}_w{4}_{5}_{6}_{7}_o{8}_{9}_{10}_{11}_ws{12}_{13}_wd{14}_{15}_p{16}_{17}_kernel0".format(
#     N, CI, H, W,
#     CO, CI, KH, KW,
#     N, CO, out_height, out_width,
#     *strides,
#     1, 1,
#     *padding
# )

# op_type = "Convolution"
# parameters = {
#     "input_shape": [N, CI, H, W], 
#     "filter_shape": [CO, CI, KH, KW], 
#     "output_shape": [N, CO, out_height, out_width],
#     "window_movement_strides": [*strides], 
#     "window_dilation_strides": [1, 1], 
#     "padding_below_diff": [*padding], 
# }

# gridDim = [v.value if not isinstance(v, int) else v for v in generator.blck_grid]
# blockDim = [v.value if not isinstance(v, int) else v for v in generator.thrd_grid]
# code = func.imported_modules[0].get_source()

# kernel = {
#     'tvm_func_name': tvm_func_name,
#     'op_type': op_type, 
#     'parameters': parameters, 
#     'code': code, 
#     'gridDim': gridDim,
#     'blockDim': blockDim
# }

# from kernel_db.convert_external import insert_kernel_entry
# insert_kernel_entry(kernel)

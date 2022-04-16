import os
import tvm
from tvm import te
import numpy as np
from scipy import signal
from tvm.contrib import nvcc
from tvm import topi
from codegen import CodeGenerator
from tvm.topi.utils import get_const_tuple
from tvm.topi.testing import depthwise_conv2d_python_nchw

@tvm.register_func("tvm_callback_cuda_compile", override=True)
def tvm_callback_cuda_compile(code):
    ptx = nvcc.compile_cuda(code, target="ptx", arch="compute_70")
    return ptx

batch = 1
in_channel = 88
in_height = 28
in_width = 28

filter_channel = in_channel
channel_multiplier = 1
filter_height = 5
filter_width = 5

stride_h = 2
stride_w = 2

padding = 2

# Placeholder
Input = te.placeholder((batch, in_channel, in_height, in_width), name="Input")
Filter = te.placeholder(
    (filter_channel, channel_multiplier, filter_height, filter_width), name="Filter"
)
Stride = [stride_h, stride_w]

# Declare
DepthwiseConv2d = topi.nn.depthwise_conv2d_nchw(Input, Filter, Stride, padding, 1)
s1 = te.create_schedule(DepthwiseConv2d.op)

# Schedule
input_np = np.random.uniform(size=get_const_tuple(Input.shape)).astype(Input.dtype)
filter_np = np.random.uniform(size=get_const_tuple(Filter.shape)).astype(Filter.dtype)

# Build the kernel
generator = CodeGenerator()
tile_dict = {"b": [1, 1], "c": [4, 2], "i": [2, 2], "j": [2, 2], "di": [1, 1], "dj": [1, 1]}
generator.rewrite_schedule(s1, tile_dict, True, True, "DepthwiseConv2d")
f1 = tvm.build(s1, [Input, Filter, DepthwiseConv2d], "cuda")
with open('depthwise-conv.cuh', 'w') as ouf:
    ouf.write('#ifndef KERNELH\n#define KERNELH\n')
    ouf.write(f1.imported_modules[0].get_source())
    ouf.write('#endif\n')

# Prepare data
ctx = tvm.gpu(0)
input_tvm = tvm.nd.array(input_np, ctx)
filter_tvm = tvm.nd.array(filter_np, ctx)

depthwise_conv2d_tvm = tvm.nd.array(
    np.zeros(shape=get_const_tuple(DepthwiseConv2d.shape), 
    dtype=DepthwiseConv2d.dtype), ctx
)

# Measure time cost of kernel 1 (depthwise_conv2d)
timer_1 = f1.time_evaluator(f1.entry_name, ctx, number=10)
tcost_1 = timer_1(input_tvm, filter_tvm, depthwise_conv2d_tvm).mean
print("average time cost of 10 runs (depthwise_conv2d) = %g us" % (tcost_1 * 1e6))

# correctness
# depthwise_conv2d_scipy = depthwise_conv2d_python_nchw(
#     input_np, filter_np, stride=[stride_h, stride_w], padding=padding
# )

# tvm.testing.assert_allclose(
#     depthwise_conv2d_tvm.asnumpy(), depthwise_conv2d_scipy, rtol=1e-5
# )
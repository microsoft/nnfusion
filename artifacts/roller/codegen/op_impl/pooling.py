import sys
import tvm
from tvm import te
import numpy as np
from scipy import signal
from tvm.contrib import nvcc
from tvm import topi
from codegen import CodeGenerator
from tvm.topi.utils import get_const_tuple
from tvm.topi.testing import adaptive_pool

@tvm.register_func("tvm_callback_cuda_compile", override=True)
def tvm_callback_cuda_compile(code):
    ptx = nvcc.compile_cuda(code, target="ptx", arch="compute_70")
    return ptx

batch   = 1
channel = 11
height  = 111
width   = 111
window  = 3
stride  = 2
padding = 1
pool_type = 'max'

if len(sys.argv) == 9:
    batch   = int(sys.argv[1])
    channel = int(sys.argv[2])
    height  = int(sys.argv[3])
    width   = int(sys.argv[4])
    window  = int(sys.argv[5])
    stride  = int(sys.argv[6])
    padding = int(sys.argv[7])
    pool_type = sys.argv[8]

# Placeholder
data = te.placeholder((batch, channel, height, width), name="data")

# Declare
pool = topi.nn.pool(data, (window, window), (stride, stride), (padding, padding, padding, padding), pool_type=pool_type)
s = te.create_schedule(pool.op)

# Schedule

# Build the kernel
generator = CodeGenerator()
tile_dict = {"ax0": [1, 1], "ax1": [1, 1], "ax2": [32, 1], "ax3": [32, 1], "dh": [1], "dw": [1]}
generator.rewrite_schedule(s, tile_dict, False, False, 'tensor')

func = tvm.build(s, [data, pool], "cuda")
with open('pool-{}.cuh'.format(
    pool_type), 'w') as ouf:
    ouf.write('#ifndef KERNELH\n#define KERNELH\n')
    ouf.write(func.imported_modules[0].get_source())
    ouf.write('#endif\n')

# Prepare data
ctx = tvm.gpu(0)
data_np = np.random.uniform(size=get_const_tuple(data.shape)).astype(data.dtype)
data_tvm = tvm.nd.array(data_np, ctx)

pooling_tvm = tvm.nd.array(
    np.zeros(shape=get_const_tuple(pool.shape), 
    dtype=pool.dtype), ctx
)

# Measure time cost of kernel 1 (pooling)
timer = func.time_evaluator(func.entry_name, ctx, number=10)
tcost = timer(data_tvm, pooling_tvm).mean
print("average time cost of 10 runs (pooling) = %g us" % (tcost * 1e6))

# correctness
out_height = (height - window + 2 * padding) // stride + 1
out_width = (width - window + 2 * padding) // stride + 1

pooling_scipy = adaptive_pool(data_np, (out_height, out_width), pool_type, 'NCHW')

tvm.testing.assert_allclose(
    pooling_tvm.asnumpy(), pooling_scipy, rtol=1e-5
)
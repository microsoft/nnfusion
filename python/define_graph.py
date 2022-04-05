import torch
import memopt
import memopt.graph as op
from memopt.utils import append_host_call, compile_and_load, compose_global_kernel
import tvm
import ctypes
import numpy as np

a = op.PlaceHolderNode("A")
b = op.PlaceHolderNode("w1")
c = op.MatMulNode([a, b], 4096, 128, 128)
d = op.PlaceHolderNode("w2")
e = op.MatMulNode([c, d], 4096, 128, 128)
f = op.OutputNode(e)

topo = op.topo_order([a, b, d])
print(topo)
config = {
    c : {'k': [8, 1], 'x': [16, 4], 'y': [16, 8]},
    e : {'k': [8, 1], 'x': [16, 4], 'y': [16, 8]},
}
target = tvm.target.cuda(arch="sm_61")
code, block_size, grid_size, args = compose_global_kernel(topo, config, target, name="Fused")
code = append_host_call(code, block_size, grid_size, 4, name="Fused", measure_time=True)
print(args)
lib = compile_and_load(code)
lib.function.restype = ctypes.c_float
torch_arrs = []
device = tvm.device(str(target), 0)
for arg in args:
    shape = list(map(int, arg.shape))
    dtype = torch.__getattribute__(arg.dtype)
    arr = torch.randn(*shape).to("cuda:0", dtype=dtype)
    torch_arrs.append(arr)

tm = lib.function(*[ctypes.c_void_p(arr.data_ptr()) for arr in torch_arrs])
assert(tm > 0)
print(tm)
ref = torch.matmul(torch.matmul(torch_arrs[0], torch_arrs[1]), torch_arrs[2])
assert(torch.max(torch.abs(ref - torch_arrs[3])) < 1e-3)

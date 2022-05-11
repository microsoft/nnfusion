import torch
import memopt
import memopt.graph as op
from memopt.utils import append_host_call, compile_and_load, compose_global_kernel
import tvm
import ctypes
import numpy as np

a = op.PlaceHolderNode("A")
b = op.PlaceHolderNode("w1")
c = op.MatMulNode([a, b], 4096, 64, 64)
d = op.PlaceHolderNode("w2")
e = op.MatMulNode([c, d], 4096, 64, 64)
f = op.OutputNode(e)

topo = op.topo_order([a, b, d])
print(topo)
config = {
    c : {'k': [16, 1], 'x': [4, 4, 2], 'y': [32, 2]},
    e : {'k': [16, 1], 'x': [4, 4, 2], 'y': [32, 2]},
}
target = tvm.target.cuda(arch="sm_61")
code, block_size, grid_size, args = compose_global_kernel(topo, config, target, name="Fused")
code = append_host_call(code, block_size, grid_size, len(args), name="Fused", measure_time=True)
print(args)
print(code)
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
print(memopt.utils.profile(lib, args))
ref = torch.matmul(torch.matmul(torch_arrs[0], torch_arrs[1]), torch_arrs[2])
assert(torch.max(torch.abs(ref - torch_arrs[3])) < 1e-3)

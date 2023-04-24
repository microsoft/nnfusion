import ctypes
import torch

from ..reference import get_ref_tensor

class _save:
    pass

def init_server(arch):
    _save.arch = arch

def close_lib(lib):
    dlclose_func = ctypes.CDLL(None).dlclose
    dlclose_func.argtypes = [ctypes.c_void_p]
    dlclose_func.restype = ctypes.c_int
    dlclose_func(lib._handle)

def call_profile(libname, args, device):
    lib = ctypes.CDLL(libname)
    lib.profile.restype = ctypes.c_float
    torch.cuda.set_device(device)

    torch_arrs = []
    for arg in args:
        shape = list(map(int, arg.shape))
        arr = get_ref_tensor(shape, device, arg.dtype)
        torch_arrs.append(arr)
    latency = lib.profile(*[ctypes.c_void_p(arr.data_ptr()) for arr in torch_arrs])
    if is_unrecoverable_error():
        # sometimes we meet unrecoverable errors like illegal memory access
        # in these cases, we have to abort and restart the profiler
        exit()
    close_lib(lib)
    if latency < 0:
        return 1e8
    return latency

def is_unrecoverable_error():
    if _save.arch.platform == "CUDA":
        cuda = ctypes.CDLL("libcudart.so")
        return cuda.cudaDeviceSynchronize() != 0
    else:
        return False

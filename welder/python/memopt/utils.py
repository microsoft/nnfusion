import ctypes
import os
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor
from typing import List

import tvm

from .config import Config
from .header import *
from .reference import get_ref_tensor
from .tvm_build import _type_map


class CompileResult:
    def __init__(self, config: Config, code: str, block_size: List[int], grid_size: List[int], name: str, args: List[tvm.te.Tensor]):
        self.config = config
        self.code = code
        self.block_size = block_size
        self.grid_size = grid_size
        self.args = args
        self.name = name
        self.lib = None
        self.latency = None
        self.origin = self
        self.use_fp16 = any([x.dtype == 'float16' for x in self.args])

    def set_io_desc(self, input_desc, output_desc):
        self.input_desc = input_desc
        self.output_desc = output_desc

    def _create_code_for_profiling(self) -> str:
        num_params = len(self.args)
        args = ["args" + str(i) for i in range(num_params)]
        call_args = ", ".join(args)
        args = ["{}* args{}".format(_type_map[self.args[i].dtype], i) for i in range(num_params)]
        def_args = ", ".join(args)
        block_str = "dim3({}, {}, {})".format(self.block_size[0], self.block_size[1], self.block_size[2])
        grid_str = "dim3({}, {}, {})".format(self.grid_size[0], self.grid_size[1], self.grid_size[2])
        call_str = "{}<<<{}, {}>>>({})".format(self.name, grid_str, block_str, call_args)
        host_funcs = \
"""
extern "C" void call({}) {{
    {};
}}
""".format(def_args, call_str)

        host_funcs += \
"""
extern "C" float profile({}) {{
    float ms;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);
    {};
    if (cudaEventRecord(stop, 0) != cudaSuccess) return -1;
    if (cudaEventSynchronize(stop) != cudaSuccess) return -1;
    if (cudaGetLastError() != cudaSuccess) return -1;
    cudaEventElapsedTime(&ms, start, stop);
    int repeats = int(ceil(100.0 / ms));
    cudaEventRecord(start, 0);
    for (int _ = 0; _ < repeats; _++)
        {};
    if (cudaEventRecord(stop, 0) != cudaSuccess) return -1;
    if (cudaEventSynchronize(stop) != cudaSuccess) return -1;
    if (cudaGetLastError() != cudaSuccess) return -1;
    cudaEventElapsedTime(&ms, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return ms / repeats;
}}
""".format(def_args, call_str, call_str)
        header = cuda_default_header + cutlass_header
        if self.use_fp16:
            header += cuda_fp16_header
        profiling_code = header + self.code + "\n" + host_funcs
        return profiling_code

    def compile_and_load(self, arch, timeout: float = None) -> ctypes.CDLL:
        if arch.platform == "CUDA":
            profiling_code = self._create_code_for_profiling()
            src = tempfile.NamedTemporaryFile(mode='w', suffix=".cu", delete=False)
            lib_name = src.name.replace(".cu", ".so")
            compute_version = arch.compute_capability
            cutlass_dir = os.path.expanduser("~/cutlass/include")
            command = ["nvcc", "--compiler-options", "'-fPIC'", "--shared", src.name, "-lcuda",
                f"-gencode=arch=compute_{compute_version},code=compute_{compute_version}",
                f"-I{cutlass_dir}", "-o", lib_name]
        elif arch.platform == "ROCm":
            profiling_code = self._create_rocm_code_for_profiling()
            src = tempfile.NamedTemporaryFile(mode='w', suffix=".cpp")
            lib_name = src.name.replace(".cpp", ".so")
            compute_version = arch.compute_capability
            command = ["hipcc", "-fPIC", "--shared", "-O2", "-ffast-math", "--amdgpu-target={}".format(compute_version),
            src.name, "-o", lib_name]
        else:
            raise NotImplementedError(arch.platform)
        src.write(profiling_code)
        src.flush()
        try:
            ret = subprocess.run(command, timeout=timeout)
        except subprocess.TimeoutExpired:
            return None
        if ret.returncode != 0:
            return None
        self.lib = ctypes.CDLL(lib_name)
        self.lib.profile.restype = ctypes.c_float
        subprocess.run(["rm", lib_name], check=True)
        return self.lib

    def _create_rocm_code_for_profiling(self) -> str:
        num_params = len(self.args)
        args = ["args" + str(i) for i in range(num_params)]
        call_args = ", ".join(args)
        args = ["{}* args{}".format(_type_map[self.args[i].dtype], i) for i in range(num_params)]
        def_args = ", ".join(args)
        block_str = "dim3({}, {}, {})".format(self.block_size[0], self.block_size[1], self.block_size[2])
        grid_str = "dim3({}, {}, {})".format(self.grid_size[0], self.grid_size[1], self.grid_size[2])
        call_str = "{}<<<{}, {}>>>({})".format(self.name, grid_str, block_str, call_args)
        host_funcs = \
"""
extern "C" void call({}) {{
    {};
}}
""".format(def_args, call_str)

        host_funcs += \
"""
extern "C" float profile({}) {{
    float ms;
    hipEvent_t start, stop;
    hipEventCreateWithFlags(&start, hipEventDefault);
    hipEventCreateWithFlags(&stop, hipEventDefault);
    hipEventRecord(start, 0);
    {};
    if (hipEventRecord(stop, 0) != hipSuccess) return -1;
    if (hipEventSynchronize(stop) != hipSuccess) return -1;
    if (hipGetLastError() != hipSuccess) return -1;
    hipEventElapsedTime(&ms, start, stop);
    int repeats = int(ceil(100.0 / ms));
    hipEventRecord(start, 0);
    for (int _ = 0; _ < repeats; _++)
        {};
    if (hipEventRecord(stop, 0) != hipSuccess) return -1;
    if (hipEventSynchronize(stop) != hipSuccess) return -1;
    if (hipGetLastError() != hipSuccess) return -1;
    hipEventElapsedTime(&ms, start, stop);
    hipEventDestroy(start);
    hipEventDestroy(stop);
    return ms / repeats;
}}
""".format(def_args, call_str, call_str)
        header = rocm_default_header
        if self.use_fp16:
            header += rocm_fp16_header
        profiling_code = header + self.code + "\n" + host_funcs
        return profiling_code

    def profile(self, device="cuda:0") -> float:
        assert self.lib
        import torch
        torch.cuda.set_device(device)
        torch_arrs = []
        for arg in self.args:
            shape = list(map(int, arg.shape))
            arr = get_ref_tensor(shape, device, arg.dtype)
            torch_arrs.append(arr)
        latency = self.lib.profile(*[ctypes.c_void_p(arr.data_ptr()) for arr in torch_arrs])
        if latency < 0:
            self.latency = 10000
            return self.latency
        self.latency = latency
        return self.latency

    def get_example_outputs(self, device="cuda:0", seed=0):
        import torch
        torch.cuda.set_device(device)
        torch.random.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch_arrs = []
        for arg in self.args:
            shape = list(map(int, arg.shape))
            arr = get_ref_tensor(shape, device, arg.dtype)
            torch_arrs.append(arr)
        self.lib.call(*[ctypes.c_void_p(arr.data_ptr()) for arr in torch_arrs])
        torch.cuda.synchronize(device)
        outputs = []
        for i, arg in enumerate(self.args):
            if arg.name.startswith("output"):
                outputs.append(torch_arrs[i].cpu().numpy())
        return outputs

    def close_lib(self):
        if self.lib is None:
            return
        dlclose_func = ctypes.CDLL(None).dlclose
        dlclose_func.argtypes = [ctypes.c_void_p]
        dlclose_func.restype = ctypes.c_int
        dlclose_func(self.lib._handle)
        self.lib = None

    def __del__(self):
        self.close_lib()

def compile_and_load_parallel(cpresults, arch, timeout : float = None):
    with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        libs = executor.map(CompileResult.compile_and_load, cpresults, [arch for _ in cpresults], [timeout for _ in cpresults])
    return list(libs)

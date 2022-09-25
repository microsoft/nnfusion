from .reference import get_ref_tensor
from .tvm_build import _type_map
from .fusion import Config
from .header import cuda_fp16_header, cuda_default_header

from typing import List
from concurrent.futures import ThreadPoolExecutor
import tvm
import ctypes
import os
import subprocess
import tempfile

class CompileResult:
    def __init__(self, config: Config, code: str, block_size: List[int], grid_size: List[int], name: str, args: List[tvm.te.Tensor]):
        self.config = config
        self.code = code
        self.block_size = block_size
        self.grid_size = grid_size
        self.args = args
        self.name = name
        self.host_code = None
        self.lib = None
        self.latency = None
        self.origin = self
        self.use_fp16 = any([x.dtype == 'float16' for x in self.args])

    def set_io_desc(self, input_desc, output_desc):
        self.input_desc = input_desc
        self.output_desc = output_desc

    def append_host_call(self) -> str:
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
        header = cuda_default_header
        if self.use_fp16:
            header += cuda_fp16_header
        self.host_code = header + self.code + "\n" + host_funcs
        return self.host_code

    def compile_and_load(self) -> ctypes.CDLL:
        assert self.host_code
        src = tempfile.NamedTemporaryFile(mode='w', suffix=".cu")
        lib_name = src.name.replace(".cu", ".so")
        src.write(self.host_code)
        src.flush()
        compute_version = "".join(tvm.contrib.nvcc.get_target_compute_version().split("."))
        ret = subprocess.run(
            ["nvcc", "--compiler-options", "'-fPIC'", "--shared", src.name, "-lcuda",
            "-gencode=arch=compute_{},code=compute_{}".format(compute_version, compute_version),
            "-o", lib_name])
        if ret.returncode != 0:
            return None
        # ret = os.system("nvcc --compiler-options '-fPIC' --shared {} -lcuda -gencode=arch=compute_61,code=compute_61 -o {}".format(src.name, lib_name))
        self.lib = ctypes.CDLL(lib_name)
        self.lib.profile.restype = ctypes.c_float
        subprocess.run(["rm", lib_name], check=True)
        return self.lib

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
            return 10000
        self.latency = latency
        return latency

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

def compile_and_load_parallel(cpresults):
    with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        libs = executor.map(CompileResult.compile_and_load, cpresults)
    return list(libs)

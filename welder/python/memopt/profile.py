import os
import subprocess
import tempfile

import numpy as np

from .header import *
from .tvm_build import _type_map
from .utils import CompileResult

cuda_profile_template = """
#include <iostream>
{header}

{kernel}

float profile({def_args}) {{
    float ms;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);
    {call_str};
    if (cudaEventRecord(stop, 0) != cudaSuccess) return -1;
    if (cudaEventSynchronize(stop) != cudaSuccess) return -1;
    if (cudaGetLastError() != cudaSuccess) return -1;
    cudaEventElapsedTime(&ms, start, stop);
    int repeats = int(ceil(1000.0 / ms));
    cudaEventRecord(start, 0);
    for (int _ = 0; _ < repeats; _++)
        {call_str};
    if (cudaEventRecord(stop, 0) != cudaSuccess) return -1;
    if (cudaEventSynchronize(stop) != cudaSuccess) return -1;
    if (cudaGetLastError() != cudaSuccess) return -1;
    cudaEventElapsedTime(&ms, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return ms / repeats;
}}

template<typename T>
__global__ void device_fill(T* ptr, size_t N) {{
  int offset = blockDim.x * blockIdx.x + threadIdx.x;
  int step = gridDim.x * blockDim.x;
  while (offset < N) {{
    ptr[offset] = T(1);
    offset += step;
  }}
}}

template<typename T>
void fillTensor(T* ptr, size_t N) {{
  dim3 grid(50, 1, 1);
  dim3 block(128, 1, 1);
  device_fill<<<grid, block>>>(ptr, N);
}}

#define deviceMalloc cudaMalloc

int main() {{
  {allocate_stmts}
  float prof = profile({call_args});
  std::cout << prof << std::endl;
  return 0;
}}
"""

rocm_profile_template = """
#include <iostream>
{header}

{kernel}

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

template<typename T>
__global__ void device_fill(T* ptr, size_t N) {{
  int offset = blockDim.x * blockIdx.x + threadIdx.x;
  int step = gridDim.x * blockDim.x;
  while (offset < N) {{
    ptr[offset] = T(1);
    offset += step;
  }}
}}

template<typename T>
void fillTensor(T* ptr, size_t N) {{
  dim3 grid(50, 1, 1);
  dim3 block(128, 1, 1);
  device_fill<<<grid, block>>>(ptr, N);
}}

#define deviceMalloc hipMalloc

int main() {{
  {allocate_stmts}
  float prof = profile({call_args});
  std::cout << prof << std::endl;
  return 0;
}}
"""

class Profiler:
    def __init__(self, cp: CompileResult) -> None:
        self.args = cp.args
        self.block_size = cp.block_size
        self.grid_size = cp.grid_size
        self.cp = cp

    def _make_profile_code(self, platform: str):
        assert platform in ["CUDA", "ROCm"]
        num_params = len(self.args)
        args = ["args" + str(i) for i in range(num_params)]
        call_args = ", ".join(args)
        args = ["{}* args{}".format(_type_map[self.args[i].dtype], i) for i in range(num_params)]
        def_args = ", ".join(args)
        block_str = "dim3({}, {}, {})".format(self.block_size[0], self.block_size[1], self.block_size[2])
        grid_str = "dim3({}, {}, {})".format(self.grid_size[0], self.grid_size[1], self.grid_size[2])
        call_str = "{}<<<{}, {}>>>({})".format(self.cp.name, grid_str, block_str, call_args)

        allocate_stmts = []
        for i in range(num_params):
            name = "args" + str(i)
            dtype = _type_map[self.args[i].dtype]
            num_elem = int(np.prod(self.args[i].shape))
            allocate_stmts.append(f"{dtype}* {name};")
            allocate_stmts.append(f"deviceMalloc(&{name}, {num_elem}*sizeof(*{name}));")
            allocate_stmts.append(f"fillTensor({name}, {num_elem});")
        allocate_stmts = "\n  ".join(allocate_stmts)
        if platform == "CUDA":
            header = cuda_default_header + cutlass_header
            template = cuda_profile_template
            if self.cp.use_fp16:
                header += cuda_fp16_header
        elif platform == "ROCm":
            header = rocm_default_header
            template = rocm_profile_template
            if self.cp.use_fp16:
                header += rocm_fp16_header
        return template.format(
            header=header,
            kernel=self.cp.code,
            def_args=def_args,
            call_str=call_str,
            allocate_stmts=allocate_stmts,
            call_args=call_args,
        )

    def _build(self, arch, timeout: float = None):
        if arch.platform == "CUDA":
            profiling_code = self._make_profile_code(arch.platform)
            src = tempfile.NamedTemporaryFile(mode='w', suffix=".cu")
            exec_name = src.name.replace(".cu", "")
            compute_version = arch.compute_capability
            cutlass_dir = os.path.expanduser("~/cutlass/include")
            command = ["nvcc", src.name, "-lcuda",
                f"-gencode=arch=compute_{compute_version},code=compute_{compute_version}",
                f"-I{cutlass_dir}", "-o", exec_name]
        elif arch.platform == "ROCm":
            profiling_code = self._make_profile_code(arch.platform)
            src = tempfile.NamedTemporaryFile(mode='w', suffix=".cpp")
            exec_name = src.name.replace(".cpp", "")
            compute_version = arch.compute_capability
            command = ["hipcc", "-O2", "-ffast-math", "--amdgpu-target={}".format(compute_version),
                src.name, "-o", exec_name]
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
        return exec_name

    def _clean_up(self, exec_name):
        subprocess.run(["rm", exec_name], check=True)

    def _run(self, exec_name, timeout: float = None):
        try:
            ret = subprocess.run([exec_name], timeout=timeout, capture_output=True)
        except subprocess.TimeoutExpired:
            return -1
        if ret.returncode != 0:
            return -1
        latency = float(ret.stdout)
        return latency

    def profile(self, arch, timeout: float = None):
        exec_name = self._build(arch, timeout)
        if exec_name is None:
            return 10000
        latency = self._run(exec_name, timeout)
        self._clean_up(exec_name)
        if latency < 0:
            return 10000
        return latency

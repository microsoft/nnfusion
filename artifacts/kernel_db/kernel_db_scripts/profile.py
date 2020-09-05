import os
import re
import subprocess
from functools import reduce


def prod(a):
    return reduce((lambda x, y: x * y), a)


def prepare_file(signature, code, config, path):
    kernel_cuh = r'''
#pragma once

#include <cuda.h>
#include <cuda_profiler_api.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <sstream>

#define CUDA_SAFE_CALL(x)                                              \
  do {                                                                 \
    cudaError_t result = (x);                                          \
    if (result != cudaSuccess) {                                       \
      const char* msg = cudaGetErrorString(result);                    \
      std::stringstream safe_call_ss;                                  \
      safe_call_ss << "\nerror: " #x " failed with error"              \
                   << "\nfile: " << __FILE__ << "\nline: " << __LINE__ \
                   << "\nmsg: " << msg;                                \
      throw std::runtime_error(safe_call_ss.str());                    \
    }                                                                  \
  } while (0)

__placeholder__

extern "C" void cuda_init() {
  int device = 0, driverVersion = 0, runtimeVersion = 0;

  CUDA_SAFE_CALL(cudaDeviceReset());
  CUDA_SAFE_CALL(cudaSetDevice(device));

  cudaDeviceProp deviceProp;
  cudaGetDeviceProperties(&deviceProp, device);
  cudaDriverGetVersion(&driverVersion);
  cudaRuntimeGetVersion(&runtimeVersion);

  printf("Profiling on Device %d: \"%s\"\n", device, deviceProp.name);
  printf("CUDA Driver Version / Runtime Version          %d.%d / %d.%d\n",
         driverVersion / 1000, (driverVersion % 100) / 10,
         runtimeVersion / 1000, (runtimeVersion % 100) / 10);
}
'''.replace("__placeholder__", code)

    profile_kernel = r'''
#include "kernel.cuh"

char* memory_pool;

int main(void) {
  cuda_init();
  CUDA_SAFE_CALL(cudaMalloc((void**)&memory_pool, __maxbytes__));
  CUDA_SAFE_CALL(cudaMemset((void*)memory_pool, 0x40, __maxbytes__));

__init_input__

  dim3 blocks__grid__;
  dim3 threads__block__;

  // warm up
  for (int i = 0; i < 5; i++) {
    __signature__<<<
        blocks, threads>>>(__input__);
  }

  // kernel call
  int steps = 100;
  cudaProfilerStart();
  for (int i_ = 0; i_ < steps; i_++) {
    __signature__<<<
        blocks, threads>>>(__input__);
  }
  cudaProfilerStop();

  CUDA_SAFE_CALL(cudaFree(memory_pool));

  return 0;
}'''
    bytes_count = [0]
    for shape in config["in_shape"]+config["out_shape"]:
        bytes_count.append(prod(shape)*4 + bytes_count[-1])
    profile_kernel = profile_kernel.replace(
        "__maxbytes__", str(bytes_count[-1]))

    init_input = ""
    input_parameters = ""
    for i in range(len(bytes_count)-1):
        init_input += "  float* arg{} = (float*)(memory_pool + {});\n".format(
            i, str(bytes_count[i]))
        input_parameters += "arg{}, ".format(i)
    input_parameters = input_parameters[:-2]
    profile_kernel = profile_kernel.replace("__init_input__", init_input)
    profile_kernel = profile_kernel.replace("__input__", input_parameters)

    profile_kernel = profile_kernel.replace(
        "__grid__", str(tuple(i for i in config["gridDim"])))
    profile_kernel = profile_kernel.replace(
        "__block__", str(tuple(i for i in config["blockDim"])))

    profile_kernel = profile_kernel.replace("__signature__", signature)

    with open(path + "kernel.cuh", "w+") as f:
        f.write(kernel_cuh)
    with open(path + "profile_kernel.cu", "w+") as f:
        f.write(profile_kernel)


def profile(kernel, path):
    command = ["make; nvprof --normalized-time-unit us --csv ./profile"]
    process = subprocess.Popen(
        command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=path, shell=True)
    device_info, nvprof = process.communicate()
    device_name = re.compile(r'Profiling on Device \d+: "(.*)"')
    kernel_profile = re.compile(
        r'"GPU activities",.+,.+,\d+,(.+),.+,.+,"{}"'.format(kernel))
    return device_name.search(str(device_info)).group(1) + ":" + kernel_profile.search(str(nvprof)).group(1) + ";"

# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
import re
import subprocess
from functools import reduce


def prod(a):
    return reduce((lambda x, y: x * y), a)


def prepare_file(signature, code, config, path, parse=False):
    profile_makefile = r'''
# Gencode arguments
# SMS ?= 30 35 37 50 52 60 61 70 75
SMS ?= 70

ifeq ($(GENCODE_FLAGS),)
# Generate SASS code for each SM architecture listed in $(SMS)
$(foreach sm,$(SMS),$(eval GENCODE_FLAGS += -gencode arch=compute_$(sm),code=sm_$(sm)))

# Generate PTX code from the highest SM architecture in $(SMS) to guarantee forward-compatibility
HIGHEST_SM := $(lastword $(sort $(SMS)))
ifneq ($(HIGHEST_SM),)
GENCODE_FLAGS += -gencode arch=compute_$(HIGHEST_SM),code=compute_$(HIGHEST_SM)
endif
endif

cc = nvcc ${GENCODE_FLAGS} -std=c++11 -I /usr/local/cuda/samples/common/inc $(LIBRARIES)

prom = profile
deps = kernel.cuh
src = $(shell find ./ -name "*.cu")
obj = $(src:%.cu=%.o)

$(prom) : $(obj)
		$(cc) -o $(prom) $(obj)

%.o : %.cu $(deps)
		$(cc) $(CFLAGS) -c $< -o $@

clean:
		rm -rf $(obj) $(prom)
'''

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
  int steps = __step__;
  cudaProfilerStart();
  for (int i_ = 0; i_ < steps; i_++) {
    __signature__<<<
        blocks, threads>>>(__input__);
  }
  cudaProfilerStop();

  CUDA_SAFE_CALL(cudaFree(memory_pool));

  return 0;
}'''.replace("__step__", "1" if parse else "100")

    if os.path.exists(path) != True:
        os.makedirs(path)

    if os.path.exists(path + "Makefile") != True:
        with open(path + "Makefile", "w+") as f:
            f.write(profile_makefile)

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


def log_sync(kernel, path):
    command = ["make; ./profile"]
    process = subprocess.Popen(
        command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=path, shell=True)
    syncthreads, _ = process.communicate()
    num_sync = re.compile(r'Amount of syncthreads logged: (\d+)')
    return int(num_sync.search(str(syncthreads)).group(1))


def profile(kernel, path):
    command = ["make; nvprof --normalized-time-unit us --csv ./profile"]
    process = subprocess.Popen(
        command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=path, shell=True)
    device_info, nvprof = process.communicate()
    device_name = re.compile(r'Profiling on Device \d+: "(.*)"')
    kernel_profile = re.compile(
        r'"GPU activities",.+,.+,\d+,(.+),.+,.+,"{}"'.format(kernel))
    return device_name.search(str(device_info)).group(1) + ":" + kernel_profile.search(str(nvprof)).group(1) + ";"

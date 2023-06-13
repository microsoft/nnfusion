import os
import re
import json


code_header = '''
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>

#ifndef __CUDA_COMMON_MACRO__
#define __CUDA_COMMON_MACRO__

#if (__CUDA_ARCH__ >= 600)

__forceinline__ __device__ __half hmax(const __half &a, const __half &b) { return a > b ? a : b; }
__forceinline__ __device__ __half hmin(const __half &a, const __half &b) { return a < b ? a : b; }

#define uint32_t unsigned

#endif

#endif
'''


def tensor_display(encoded_name, prop):
  return f'{encoded_name}:{prop["dtype"]}{str(prop["shape"])}'


def kernel_slice_to_code(kernel_slice, kernel_name, in_args, out_args, thread_extent):
  # insert thread extent info
  thread_extent_suffix = ['.x', '.y', '.z']
  grid_size = ['  // [thread_extent] blockIdx%s = %d' % (sfx, val) for sfx, val in zip(thread_extent_suffix, thread_extent['grid_size'])]
  block_size = ['  // [thread_extent] threadIdx%s = %d' % (sfx, val) for sfx, val in zip(thread_extent_suffix, thread_extent['block_size'])]
  idx = kernel_slice.find('__global__')
  idx = kernel_slice.find(') {\n', idx) + 4
  kernel_slice = kernel_slice[:idx] + '\n'.join(grid_size + block_size) +'\n' + kernel_slice[idx:]
  # add code header
  kernel = code_header + '\n' + kernel_slice
  display_inputs = ', '.join([tensor_display(name, prop) for (name, prop) in in_args])
  display_outputs = ', '.join([tensor_display(name, prop) for (name, prop) in out_args])
  code = f'// LOCAL: {kernel_name} -- {display_inputs} -> {display_outputs}\n\n{kernel}\n'
  return code


def pack_kernel_slices(kernel_slices):
  code = ['']
  for slice, name, in_args, out_args, thread_extent in kernel_slices:
    code.append(kernel_slice_to_code(slice, name, in_args, out_args, thread_extent))
  code = '\n// ---------------------------------------------------------------------------\n'.join(code)
  return code


def get_kernel_metadata(exprss, global_input_list, global_output_list, config = 'null', backend = 'c-cuda', device_code = 'default'):
  inp_args, outp_args = [], []
  for name, prop in global_input_list:
    inp_args.append('%s:%s%s' % (name, prop['dtype'], prop['shape']))
  for name, prop in global_output_list:
    outp_args.append('%s:%s%s' % (name, prop['dtype'], prop['shape']))
  header_meta = '// GLOBALS: ' + ', '.join(inp_args) + ' -> ' + ', '.join(outp_args) + '\n// BACKEND: %s (%s)\n' % (backend, device_code)
  properties = "// CONFIG: %s\n// COMPUTE_V1: %s\n" % (config.strip() if isinstance(config, str) else '', exprss)
  return header_meta + properties



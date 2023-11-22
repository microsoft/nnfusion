import os
import re
import json
from common_header import *

code_header = '''
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>

#ifndef __CUDA_COMMON_MACRO__
#define __CUDA_COMMON_MACRO__

__device__ half max(half a, half b)
{
  return __hgt(__half(a), __half(b)) ? a : b;
}
__device__ half min(half a, half b)
{
  return __hlt(__half(a), __half(b)) ? a : b;
}

typedef long long _ll;
#define int64_t _ll
#define __int8_t_defined

#define CUDA_UNSUPPORTED_HALF_MATH_BINARY(HALF_MATH_NAME, FP32_MATH_NAME) inline __device__ half HALF_MATH_NAME(half x, half y) {     float tmp_x = __half2float(x);                                            float tmp_y = __half2float(y);                                            float result = FP32_MATH_NAME(tmp_x, tmp_y);                              return __float2half(result);                                            }

#define CUDA_UNSUPPORTED_HALF_MATH_UNARY(HALF_MATH_NAME, FP32_MATH_NAME) inline __device__ half HALF_MATH_NAME(half x) {            float tmp_x = __half2float(x);                                           float result = FP32_MATH_NAME(tmp_x);                                    return __float2half(result);                                           }

CUDA_UNSUPPORTED_HALF_MATH_BINARY(hpow, powf)
CUDA_UNSUPPORTED_HALF_MATH_UNARY(htanh, tanhf)
CUDA_UNSUPPORTED_HALF_MATH_UNARY(htan, tanf)
CUDA_UNSUPPORTED_HALF_MATH_UNARY(hatan, atanf)
CUDA_UNSUPPORTED_HALF_MATH_UNARY(herf, erf)

#undef CUDA_UNSUPPORTED_HALF_MATH_BINARY
#undef CUDA_UNSUPPORTED_HALF_MATH_UNARY

// Pack two half values.
inline __device__ __host__ unsigned
__pack_half2(const half x, const half y) {
  unsigned v0 = *((unsigned short *)&x);
  unsigned v1 = *((unsigned short *)&y);
  return (v1 << 16) | v0;
}

// There is no make_int8 in cuda, but TVM codegen seem to use it
inline __device__ longlong4 make_int8(int x0, int x1, int x2, int x3, int x4, int x5, int x6, int x7) {
  int2 i0 = make_int2(x0, x1);
  int2 i1 = make_int2(x2, x3);
  int2 i2 = make_int2(x4, x5);
  int2 i3 = make_int2(x6, x7);
  long long l0 = *(long long*)&i0;
  long long l1 = *(long long*)&i1;
  long long l2 = *(long long*)&i2;
  long long l3 = *(long long*)&i3;
  return make_longlong4(l0, l1, l2, l3);
}


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
  idx1 = kernel_slice.find('__global__')
  idx2 = kernel_slice.find(') {\n', idx1) + 4
  kernel_slice = kernel_slice[:idx1] + 'extern "C" ' + kernel_slice[idx1 : idx2] + '\n'.join(grid_size + block_size) +'\n' + kernel_slice[idx2:]
  # add code header
  header = cuda_default_header
  if re.search('cutlass', kernel_slice):
     header += cutlass_header
  if re.search('half', kernel_slice):
    header += cuda_fp16_header
  kernel = header + '\n' + kernel_slice
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



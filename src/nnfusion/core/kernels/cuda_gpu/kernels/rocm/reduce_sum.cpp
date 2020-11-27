// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "../../cuda_emitter.hpp"
#include "../../cuda_langunit.hpp"
#include "nnfusion/core/operators/generic_op/generic_op.hpp"

namespace nnfusion
{
    namespace kernels
    {
        namespace cuda
        {
            template <class T>
            class RocmReduce : public CudaEmitter
            {
            public:
                RocmReduce(shared_ptr<KernelContext> ctx)
                    : CudaEmitter(ctx)
                {
                }

                LanguageUnit_p put_source(const std::string& src, bool update_config = false)
                {
                    LanguageUnit_p _lu(new LanguageUnit(get_function_name()));
                    auto& lu = *_lu;
                    lu << src;
                    if (update_config)
                    {
                        int at_bx = src.find("// [thread_extent] blockIdx.x = "),
                            blockX = (at_bx >= 0)
                                         ? atoi(src.c_str() + at_bx +
                                                sizeof("// [thread_extent] blockIdx.x = ") - 1)
                                         : 1;
                        int at_by = src.find("// [thread_extent] blockIdx.y = "),
                            blockY = (at_by >= 0)
                                         ? atoi(src.c_str() + at_by +
                                                sizeof("// [thread_extent] blockIdx.y = ") - 1)
                                         : 1;
                        int at_bz = src.find("// [thread_extent] blockIdx.z = "),
                            blockZ = (at_bz >= 0)
                                         ? atoi(src.c_str() + at_bz +
                                                sizeof("// [thread_extent] blockIdx.z = ") - 1)
                                         : 1;
                        int at_tx = src.find("// [thread_extent] threadIdx.x = "),
                            threadX = (at_tx >= 0)
                                          ? atoi(src.c_str() + at_tx +
                                                 sizeof("// [thread_extent] threadIdx.x = ") - 1)
                                          : 1;
                        int at_ty = src.find("// [thread_extent] threadIdx.y = "),
                            threadY = (at_ty >= 0)
                                          ? atoi(src.c_str() + at_ty +
                                                 sizeof("// [thread_extent] threadIdx.y = ") - 1)
                                          : 1;
                        int at_tz = src.find("// [thread_extent] threadIdx.z = "),
                            threadZ = (at_tz >= 0)
                                          ? atoi(src.c_str() + at_tz +
                                                 sizeof("// [thread_extent] threadIdx.z = ") - 1)
                                          : 1;

                        m_gridDim = dim3(blockX, blockY, blockZ);
                        m_blockDim = dim3(threadX, threadY, threadZ);
                    }
                    return _lu;
                }

                LanguageUnit_p emit_function_body() override
                {
                    auto& ctx = m_context;
                    auto _op = dynamic_pointer_cast<nnfusion::op::Sum>(ctx->gnode->get_op_ptr());

                    auto input_shape = nnfusion::Shape(ctx->inputs[0]->get_shape());
                    auto output_shape = nnfusion::Shape(ctx->outputs[0]->get_shape());
                    auto reduce_axis = _op->get_reduction_axes();

                    int min_axis = input_shape.size(), max_axis = -1, reduce_scale = 1;
                    for (auto& axis : reduce_axis)
                    {
                        min_axis = min(min_axis, (int)axis);
                        max_axis = max(max_axis, (int)axis);
                        reduce_scale *= input_shape[axis];
                    }
                    size_t tensor_size = std::accumulate(
                        input_shape.begin(), input_shape.end(), 1LU, std::multiplies<int>());
                    if (reduce_scale == 1 || min_axis > max_axis) // as memcpy
                    {
                        return nullptr; // using cuda's inplace solution

                        int blocks = tensor_size, threads = 1;
                        for (int i = 1024; i > 1; --i)
                        {
                            if (tensor_size % i == 0)
                            {
                                threads = i;
                                blocks = tensor_size / i;
                                break;
                            }
                        }
                        m_gridDim = dim3(blocks, 1, 1);
                        m_blockDim = dim3(threads, 1, 1);

                        return put_source(nnfusion::op::create_code_from_template(
                            R"(output0[((int)blockIdx.x) * @num_threads@ + ((int)threadIdx.x)] = input0[((int)blockIdx.x) * @num_threads@ + ((int)threadIdx.x)];)",
                            {{"num_threads", threads}}));
                    }
                    else // ReduceSum 2D
                    {
                        int groups, samples, stride_group, stride_sample;
                        if (min_axis == 0 && max_axis == reduce_axis.size() - 1) // A[X][Y] -> B[Y]
                        {
                            samples = std::accumulate(input_shape.begin(),
                                                      input_shape.begin() + reduce_axis.size(),
                                                      1LU,
                                                      std::multiplies<int>());
                            groups = tensor_size / samples;
                            stride_group = 1;
                            stride_sample = groups;
                        }
                        else if (min_axis == input_shape.size() - reduce_axis.size() &&
                                 max_axis == input_shape.size() - 1) // A[X][Y] -> B[X]
                        {
                            samples = std::accumulate(input_shape.end() - reduce_axis.size(),
                                                      input_shape.end(),
                                                      1LU,
                                                      std::multiplies<int>());
                            groups = tensor_size / samples;
                            stride_group = samples;
                            stride_sample = 1;
                        }
                        else
                            return nullptr;

                        if (groups == 1024 && samples == 3072 && stride_group == 1)
                        {
                            std::string src = R"(
  // [thread_extent] blockIdx.x = 64
   float output0_local[1];
  // [thread_extent] threadIdx.x = 16
  output0_local[0] = 0.000000e+00f;
  for (int k_outer = 0; k_outer < 96; ++k_outer) {
    for (int k_inner_inner = 0; k_inner_inner < 2; ++k_inner_inner) {
      output0_local[0] = (output0_local[0] + input0[((((k_outer * 32768) + (k_inner_inner * 1024)) + (((int)blockIdx.x) * 16)) + ((int)threadIdx.x))]);
      output0_local[0] = (output0_local[0] + input0[(((((k_outer * 32768) + (k_inner_inner * 1024)) + (((int)blockIdx.x) * 16)) + ((int)threadIdx.x)) + 2048)]);
      output0_local[0] = (output0_local[0] + input0[(((((k_outer * 32768) + (k_inner_inner * 1024)) + (((int)blockIdx.x) * 16)) + ((int)threadIdx.x)) + 4096)]);
      output0_local[0] = (output0_local[0] + input0[(((((k_outer * 32768) + (k_inner_inner * 1024)) + (((int)blockIdx.x) * 16)) + ((int)threadIdx.x)) + 6144)]);
      output0_local[0] = (output0_local[0] + input0[(((((k_outer * 32768) + (k_inner_inner * 1024)) + (((int)blockIdx.x) * 16)) + ((int)threadIdx.x)) + 8192)]);
      output0_local[0] = (output0_local[0] + input0[(((((k_outer * 32768) + (k_inner_inner * 1024)) + (((int)blockIdx.x) * 16)) + ((int)threadIdx.x)) + 10240)]);
      output0_local[0] = (output0_local[0] + input0[(((((k_outer * 32768) + (k_inner_inner * 1024)) + (((int)blockIdx.x) * 16)) + ((int)threadIdx.x)) + 12288)]);
      output0_local[0] = (output0_local[0] + input0[(((((k_outer * 32768) + (k_inner_inner * 1024)) + (((int)blockIdx.x) * 16)) + ((int)threadIdx.x)) + 14336)]);
      output0_local[0] = (output0_local[0] + input0[(((((k_outer * 32768) + (k_inner_inner * 1024)) + (((int)blockIdx.x) * 16)) + ((int)threadIdx.x)) + 16384)]);
      output0_local[0] = (output0_local[0] + input0[(((((k_outer * 32768) + (k_inner_inner * 1024)) + (((int)blockIdx.x) * 16)) + ((int)threadIdx.x)) + 18432)]);
      output0_local[0] = (output0_local[0] + input0[(((((k_outer * 32768) + (k_inner_inner * 1024)) + (((int)blockIdx.x) * 16)) + ((int)threadIdx.x)) + 20480)]);
      output0_local[0] = (output0_local[0] + input0[(((((k_outer * 32768) + (k_inner_inner * 1024)) + (((int)blockIdx.x) * 16)) + ((int)threadIdx.x)) + 22528)]);
      output0_local[0] = (output0_local[0] + input0[(((((k_outer * 32768) + (k_inner_inner * 1024)) + (((int)blockIdx.x) * 16)) + ((int)threadIdx.x)) + 24576)]);
      output0_local[0] = (output0_local[0] + input0[(((((k_outer * 32768) + (k_inner_inner * 1024)) + (((int)blockIdx.x) * 16)) + ((int)threadIdx.x)) + 26624)]);
      output0_local[0] = (output0_local[0] + input0[(((((k_outer * 32768) + (k_inner_inner * 1024)) + (((int)blockIdx.x) * 16)) + ((int)threadIdx.x)) + 28672)]);
      output0_local[0] = (output0_local[0] + input0[(((((k_outer * 32768) + (k_inner_inner * 1024)) + (((int)blockIdx.x) * 16)) + ((int)threadIdx.x)) + 30720)]);
    }
  }
  output0[((((int)blockIdx.x) * 16) + ((int)threadIdx.x))] = output0_local[0];
#if 0
  // [thread_extent] blockIdx.x = 64
   float output0_local[1];
  // [thread_extent] threadIdx.x = 16
  output0_local[0] = 0.000000e+00f;
  for (int k_outer = 0; k_outer < 64; ++k_outer) {
    output0_local[0] = (output0_local[0] + input0[(((k_outer * 49152) + (((int)blockIdx.x) * 16)) + ((int)threadIdx.x))]);
    output0_local[0] = (output0_local[0] + input0[((((k_outer * 49152) + (((int)blockIdx.x) * 16)) + ((int)threadIdx.x)) + 1024)]);
    output0_local[0] = (output0_local[0] + input0[((((k_outer * 49152) + (((int)blockIdx.x) * 16)) + ((int)threadIdx.x)) + 2048)]);
    output0_local[0] = (output0_local[0] + input0[((((k_outer * 49152) + (((int)blockIdx.x) * 16)) + ((int)threadIdx.x)) + 3072)]);
    output0_local[0] = (output0_local[0] + input0[((((k_outer * 49152) + (((int)blockIdx.x) * 16)) + ((int)threadIdx.x)) + 4096)]);
    output0_local[0] = (output0_local[0] + input0[((((k_outer * 49152) + (((int)blockIdx.x) * 16)) + ((int)threadIdx.x)) + 5120)]);
    output0_local[0] = (output0_local[0] + input0[((((k_outer * 49152) + (((int)blockIdx.x) * 16)) + ((int)threadIdx.x)) + 6144)]);
    output0_local[0] = (output0_local[0] + input0[((((k_outer * 49152) + (((int)blockIdx.x) * 16)) + ((int)threadIdx.x)) + 7168)]);
    output0_local[0] = (output0_local[0] + input0[((((k_outer * 49152) + (((int)blockIdx.x) * 16)) + ((int)threadIdx.x)) + 8192)]);
    output0_local[0] = (output0_local[0] + input0[((((k_outer * 49152) + (((int)blockIdx.x) * 16)) + ((int)threadIdx.x)) + 9216)]);
    output0_local[0] = (output0_local[0] + input0[((((k_outer * 49152) + (((int)blockIdx.x) * 16)) + ((int)threadIdx.x)) + 10240)]);
    output0_local[0] = (output0_local[0] + input0[((((k_outer * 49152) + (((int)blockIdx.x) * 16)) + ((int)threadIdx.x)) + 11264)]);
    output0_local[0] = (output0_local[0] + input0[((((k_outer * 49152) + (((int)blockIdx.x) * 16)) + ((int)threadIdx.x)) + 12288)]);
    output0_local[0] = (output0_local[0] + input0[((((k_outer * 49152) + (((int)blockIdx.x) * 16)) + ((int)threadIdx.x)) + 13312)]);
    output0_local[0] = (output0_local[0] + input0[((((k_outer * 49152) + (((int)blockIdx.x) * 16)) + ((int)threadIdx.x)) + 14336)]);
    output0_local[0] = (output0_local[0] + input0[((((k_outer * 49152) + (((int)blockIdx.x) * 16)) + ((int)threadIdx.x)) + 15360)]);
    output0_local[0] = (output0_local[0] + input0[((((k_outer * 49152) + (((int)blockIdx.x) * 16)) + ((int)threadIdx.x)) + 16384)]);
    output0_local[0] = (output0_local[0] + input0[((((k_outer * 49152) + (((int)blockIdx.x) * 16)) + ((int)threadIdx.x)) + 17408)]);
    output0_local[0] = (output0_local[0] + input0[((((k_outer * 49152) + (((int)blockIdx.x) * 16)) + ((int)threadIdx.x)) + 18432)]);
    output0_local[0] = (output0_local[0] + input0[((((k_outer * 49152) + (((int)blockIdx.x) * 16)) + ((int)threadIdx.x)) + 19456)]);
    output0_local[0] = (output0_local[0] + input0[((((k_outer * 49152) + (((int)blockIdx.x) * 16)) + ((int)threadIdx.x)) + 20480)]);
    output0_local[0] = (output0_local[0] + input0[((((k_outer * 49152) + (((int)blockIdx.x) * 16)) + ((int)threadIdx.x)) + 21504)]);
    output0_local[0] = (output0_local[0] + input0[((((k_outer * 49152) + (((int)blockIdx.x) * 16)) + ((int)threadIdx.x)) + 22528)]);
    output0_local[0] = (output0_local[0] + input0[((((k_outer * 49152) + (((int)blockIdx.x) * 16)) + ((int)threadIdx.x)) + 23552)]);
    output0_local[0] = (output0_local[0] + input0[((((k_outer * 49152) + (((int)blockIdx.x) * 16)) + ((int)threadIdx.x)) + 24576)]);
    output0_local[0] = (output0_local[0] + input0[((((k_outer * 49152) + (((int)blockIdx.x) * 16)) + ((int)threadIdx.x)) + 25600)]);
    output0_local[0] = (output0_local[0] + input0[((((k_outer * 49152) + (((int)blockIdx.x) * 16)) + ((int)threadIdx.x)) + 26624)]);
    output0_local[0] = (output0_local[0] + input0[((((k_outer * 49152) + (((int)blockIdx.x) * 16)) + ((int)threadIdx.x)) + 27648)]);
    output0_local[0] = (output0_local[0] + input0[((((k_outer * 49152) + (((int)blockIdx.x) * 16)) + ((int)threadIdx.x)) + 28672)]);
    output0_local[0] = (output0_local[0] + input0[((((k_outer * 49152) + (((int)blockIdx.x) * 16)) + ((int)threadIdx.x)) + 29696)]);
    output0_local[0] = (output0_local[0] + input0[((((k_outer * 49152) + (((int)blockIdx.x) * 16)) + ((int)threadIdx.x)) + 30720)]);
    output0_local[0] = (output0_local[0] + input0[((((k_outer * 49152) + (((int)blockIdx.x) * 16)) + ((int)threadIdx.x)) + 31744)]);
    output0_local[0] = (output0_local[0] + input0[((((k_outer * 49152) + (((int)blockIdx.x) * 16)) + ((int)threadIdx.x)) + 32768)]);
    output0_local[0] = (output0_local[0] + input0[((((k_outer * 49152) + (((int)blockIdx.x) * 16)) + ((int)threadIdx.x)) + 33792)]);
    output0_local[0] = (output0_local[0] + input0[((((k_outer * 49152) + (((int)blockIdx.x) * 16)) + ((int)threadIdx.x)) + 34816)]);
    output0_local[0] = (output0_local[0] + input0[((((k_outer * 49152) + (((int)blockIdx.x) * 16)) + ((int)threadIdx.x)) + 35840)]);
    output0_local[0] = (output0_local[0] + input0[((((k_outer * 49152) + (((int)blockIdx.x) * 16)) + ((int)threadIdx.x)) + 36864)]);
    output0_local[0] = (output0_local[0] + input0[((((k_outer * 49152) + (((int)blockIdx.x) * 16)) + ((int)threadIdx.x)) + 37888)]);
    output0_local[0] = (output0_local[0] + input0[((((k_outer * 49152) + (((int)blockIdx.x) * 16)) + ((int)threadIdx.x)) + 38912)]);
    output0_local[0] = (output0_local[0] + input0[((((k_outer * 49152) + (((int)blockIdx.x) * 16)) + ((int)threadIdx.x)) + 39936)]);
    output0_local[0] = (output0_local[0] + input0[((((k_outer * 49152) + (((int)blockIdx.x) * 16)) + ((int)threadIdx.x)) + 40960)]);
    output0_local[0] = (output0_local[0] + input0[((((k_outer * 49152) + (((int)blockIdx.x) * 16)) + ((int)threadIdx.x)) + 41984)]);
    output0_local[0] = (output0_local[0] + input0[((((k_outer * 49152) + (((int)blockIdx.x) * 16)) + ((int)threadIdx.x)) + 43008)]);
    output0_local[0] = (output0_local[0] + input0[((((k_outer * 49152) + (((int)blockIdx.x) * 16)) + ((int)threadIdx.x)) + 44032)]);
    output0_local[0] = (output0_local[0] + input0[((((k_outer * 49152) + (((int)blockIdx.x) * 16)) + ((int)threadIdx.x)) + 45056)]);
    output0_local[0] = (output0_local[0] + input0[((((k_outer * 49152) + (((int)blockIdx.x) * 16)) + ((int)threadIdx.x)) + 46080)]);
    output0_local[0] = (output0_local[0] + input0[((((k_outer * 49152) + (((int)blockIdx.x) * 16)) + ((int)threadIdx.x)) + 47104)]);
    output0_local[0] = (output0_local[0] + input0[((((k_outer * 49152) + (((int)blockIdx.x) * 16)) + ((int)threadIdx.x)) + 48128)]);
  }
  output0[((((int)blockIdx.x) * 16) + ((int)threadIdx.x))] = output0_local[0];
#endif
)";
                            return put_source(src, true);
                        }
                        if (groups == 4096 && samples == 3072 && stride_group == 1)
                        {
                            std::string src = R"(
  // [thread_extent] blockIdx.x = 256
   float output0_local[1];
  // [thread_extent] threadIdx.x = 16
  output0_local[0] = 0.000000e+00f;
  for (int k_outer = 0; k_outer < 384; ++k_outer) {
    output0_local[0] = (output0_local[0] + input0[(((k_outer * 32768) + (((int)blockIdx.x) * 16)) + ((int)threadIdx.x))]);
    output0_local[0] = (output0_local[0] + input0[((((k_outer * 32768) + (((int)blockIdx.x) * 16)) + ((int)threadIdx.x)) + 4096)]);
    output0_local[0] = (output0_local[0] + input0[((((k_outer * 32768) + (((int)blockIdx.x) * 16)) + ((int)threadIdx.x)) + 8192)]);
    output0_local[0] = (output0_local[0] + input0[((((k_outer * 32768) + (((int)blockIdx.x) * 16)) + ((int)threadIdx.x)) + 12288)]);
    output0_local[0] = (output0_local[0] + input0[((((k_outer * 32768) + (((int)blockIdx.x) * 16)) + ((int)threadIdx.x)) + 16384)]);
    output0_local[0] = (output0_local[0] + input0[((((k_outer * 32768) + (((int)blockIdx.x) * 16)) + ((int)threadIdx.x)) + 20480)]);
    output0_local[0] = (output0_local[0] + input0[((((k_outer * 32768) + (((int)blockIdx.x) * 16)) + ((int)threadIdx.x)) + 24576)]);
    output0_local[0] = (output0_local[0] + input0[((((k_outer * 32768) + (((int)blockIdx.x) * 16)) + ((int)threadIdx.x)) + 28672)]);
  }
  output0[((((int)blockIdx.x) * 16) + ((int)threadIdx.x))] = output0_local[0];
)";
                            return put_source(src, true);
                        }
                        if (groups == 3072 && samples == 1024 && stride_group != 1)
                        {
                            std::string src = R"(
  // [thread_extent] blockIdx.x = 192
   float output0_local[1];
  // [thread_extent] threadIdx.x = 16
  output0_local[0] = 0.000000e+00f;
  for (int k_outer = 0; k_outer < 32; ++k_outer) {
    for (int k_inner_inner = 0; k_inner_inner < 8; ++k_inner_inner) {
      output0_local[0] = (output0_local[0] + input0[((((((int)blockIdx.x) * 16384) + (((int)threadIdx.x) * 1024)) + (k_outer * 32)) + k_inner_inner)]);
      output0_local[0] = (output0_local[0] + input0[(((((((int)blockIdx.x) * 16384) + (((int)threadIdx.x) * 1024)) + (k_outer * 32)) + k_inner_inner) + 8)]);
      output0_local[0] = (output0_local[0] + input0[(((((((int)blockIdx.x) * 16384) + (((int)threadIdx.x) * 1024)) + (k_outer * 32)) + k_inner_inner) + 16)]);
      output0_local[0] = (output0_local[0] + input0[(((((((int)blockIdx.x) * 16384) + (((int)threadIdx.x) * 1024)) + (k_outer * 32)) + k_inner_inner) + 24)]);
    }
  }
  output0[((((int)blockIdx.x) * 16) + ((int)threadIdx.x))] = output0_local[0];
)";
                            return put_source(src, true);
                        }
                        if (groups == 1024 * 512 && samples == 6 && stride_group == 1)
                        {
                            std::string src = R"(
  // [thread_extent] blockIdx.x = 1024
   float output0_local[8];
  // [thread_extent] threadIdx.x = 64
  for (int i_c_init = 0; i_c_init < 8; ++i_c_init) {
    output0_local[i_c_init] = 0.000000e+00f;
  }
  for (int k_outer = 0; k_outer < 3; ++k_outer) {
    for (int k_inner_inner = 0; k_inner_inner < 2; ++k_inner_inner) {
      for (int i_c = 0; i_c < 8; ++i_c) {
        output0_local[i_c] = (output0_local[i_c] + input0[(((((k_outer * 1048576) + (k_inner_inner * 524288)) + (((int)blockIdx.x) * 512)) + (((int)threadIdx.x) * 8)) + i_c)]);
      }
    }
  }
  for (int i_inner_inner_inner = 0; i_inner_inner_inner < 8; ++i_inner_inner_inner) {
    output0[(((((int)blockIdx.x) * 512) + (((int)threadIdx.x) * 8)) + i_inner_inner_inner)] = output0_local[i_inner_inner_inner];
  }
)";
                            return put_source(src, true);
                        }
                        int numThreads = 4; // tunable: 2, 4, 8, 16, 32, 64, 128, 512, 1024

                        m_gridDim = dim3(1, groups, 1);
                        m_blockDim = dim3(numThreads, 1, 1);

                        return put_source(nnfusion::op::create_code_from_template(
                            R"(
    constexpr int numThreads = @num_threads@;
    extern __shared__ float Isdata[numThreads];
    int tid = threadIdx.x;
    Isdata[tid] = 0;

    int i = tid;
    #pragma unroll
    while (i < @samples@) { Isdata[tid] += input0[i * @stride_sample@ + ((int)blockIdx.y) * @stride_group@]; i += numThreads; }
    if (numThreads >= 128) { __syncthreads(); }

    if (numThreads >= 512) { if (tid < 256) { Isdata[tid] += Isdata[tid + 256]; } __syncthreads(); }
    if (numThreads >= 256) { if (tid < 128) { Isdata[tid] += Isdata[tid + 128]; } __syncthreads(); }

    volatile float *__sdata = (volatile float *)Isdata;
    if (numThreads >= 128) __sdata[tid] += __sdata[tid + 64], __syncthreads();
    if (numThreads >= 64) __sdata[tid] += __sdata[tid + 32], __syncthreads();
    if (numThreads >= 32) __sdata[tid] += __sdata[tid + 16], __syncthreads();
    if (numThreads >= 16) __sdata[tid] += __sdata[tid + 8], __syncthreads();
    if (numThreads >= 8) __sdata[tid] += __sdata[tid + 4], __syncthreads();
    if (numThreads >= 4) __sdata[tid] += __sdata[tid + 2], __syncthreads();
    if (numThreads >= 2) __sdata[tid] += __sdata[tid + 1], __syncthreads();
    if (tid == 0) output0[((int)blockIdx.y)] = Isdata[0];
)",
                            {{"samples", samples},
                             {"stride_sample", stride_sample},
                             {"stride_group", stride_group},
                             {"num_threads", numThreads}}));
                    }
                }

                LanguageUnit_p emit_dependency() override
                {
                    LanguageUnit_p _lu(new LanguageUnit(get_function_name() + "_dep"));
                    _lu->require(header::cuda);
                    return _lu;
                }

                void set_launch_config() override {}
            };
        } // namespace cuda
    }     // namespace kernels
} // namespace nnfusion

using namespace nnfusion;
using namespace nnfusion::kernels;

#define REGISTER_GPU_KERNEL(KEY, OP_NAME)                                                          \
    REGISTER_KERNEL_EMITTER(KEY,                                                                   \
                            Device(ROCM_GPU).TypeConstraint(element::f32).Priority(4),             \
                            cuda::RocmReduce<nnfusion::op::OP_NAME>)

REGISTER_GPU_KERNEL("Sum", Add)

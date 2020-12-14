// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "../../cuda_emitter.hpp"
#include "../../cuda_langunit.hpp"
#include "nnfusion/core/operators/generic_op/generic_op.hpp"

DECLARE_bool(frocm_fixed_kernels);

namespace nnfusion
{
    namespace kernels
    {
        namespace cuda
        {
            class GemmFixed : public CudaEmitter
            {
                shared_ptr<nnfusion::op::GenericOp> generic_op;

            public:
                GemmFixed(shared_ptr<KernelContext> ctx)
                    : CudaEmitter(ctx)
                    , generic_op(
                          static_pointer_cast<nnfusion::op::GenericOp>(ctx->gnode->get_op_ptr()))
                {
                    GENERIC_OP_LOGGING();
                }

                LanguageUnit_p emit_function_body() override
                {
                    bool using_fixed = FLAGS_frocm_fixed_kernels;
                    if (!using_fixed)
                        return nullptr;

                    GENERIC_OP_LOGGING();
                    auto& ctx = m_context;

                    auto& arg0_shape = ctx->inputs[0]->get_shape();
                    auto& arg1_shape = ctx->inputs[1]->get_shape();
                    auto& out_shape = ctx->outputs[0]->get_shape();

                    auto gemm = static_pointer_cast<nnfusion::op::Dot>(ctx->gnode->get_op_ptr());
                    auto reduction_axes = gemm->get_reduction_axes_count();
                    auto& dtype = ctx->outputs[0]->get_element_type().c_type_string();
                    if (gemm->get_transpose_A())
                        return nullptr;
                    auto transpose_B = gemm->get_transpose_B();

                    if (arg0_shape.empty() || arg1_shape.empty())
                        return nullptr;
                    if ((arg0_shape.size() == arg1_shape.size()) &&
                        (arg0_shape.size() == reduction_axes))
                        return nullptr;
                    if ((arg0_shape.size() == 2) && (arg1_shape.size() == 1) &&
                        (reduction_axes == 1))
                        return nullptr;
                    if (dtype != "float")
                        return nullptr;
                    return nullptr;

                    std::string templ;
                    if (arg0_shape[0] == 1)
                    {
                        if (arg0_shape[1] > 1024)
                            return nullptr;
                        m_gridDim = dim3(1, 1, 1);
                        m_blockDim = dim3(64, 1, 1);

                        int dataSize;

                        if (transpose_B)
                        {
                            if (arg0_shape[1] != arg1_shape[1])
                                return nullptr;
                            m_gridDim.y = arg1_shape[0];
                        }
                        else
                        {
                            if (arg0_shape[1] != arg1_shape[0])
                                return nullptr;
                            m_gridDim.y = arg1_shape[1];
                        }

                        templ = nnfusion::op::create_code_from_template(
                            R"(
	constexpr unsigned int gridDimX = @gridDimX@;
    constexpr unsigned int gridDimY = @gridDimY@;
    constexpr unsigned int blockSize = @blockSize@;
    constexpr unsigned int dataSize = @dataSize@;
    constexpr bool transpose_B = @transpose_B@;

    extern __shared__ float sdata[blockSize];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x*(blockSize*2) + tid;
    constexpr unsigned int gridSize = blockSize*2*gridDimX;
    sdata[tid] = 0;
    if (transpose_B) {
        #pragma unroll
        while (i < dataSize) { sdata[tid] += input0[i] * input1[blockIdx.y * dataSize + i] + input0[i + blockSize] * input1[blockIdx.y * dataSize + i + blockSize]; i += gridSize; }
    } else {
        #pragma unroll
        while (i < dataSize) { sdata[tid] += input0[i] * input1[blockIdx.y + gridDimY * i] + input0[i + blockSize] * input1[blockIdx.y + gridDimY * i + gridDimY * blockSize]; i += gridSize; }
    }
    if (blockSize >= 128) { __syncthreads(); }
    if (blockSize >= 512) { if (tid < 256) { sdata[tid] += sdata[tid + 256]; } __syncthreads(); }
    if (blockSize >= 256) { if (tid < 128) { sdata[tid] += sdata[tid + 128]; } __syncthreads(); }

    // warp_reduce
    volatile float *__sdata = (volatile float *)sdata;
    if (blockSize >= 128) __sdata[tid] += __sdata[tid + 64];
    if (blockSize >= 64) __sdata[tid] += __sdata[tid + 32];
    if (blockSize >= 32) __sdata[tid] += __sdata[tid + 16];
    if (blockSize >= 16) __sdata[tid] += __sdata[tid + 8];
    if (blockSize >= 8) __sdata[tid] += __sdata[tid + 4];
    if (blockSize >= 4) __sdata[tid] += __sdata[tid + 2];
    if (blockSize >= 2) __sdata[tid] += __sdata[tid + 1];

    if (tid == 0) output0[blockIdx.y] = sdata[0];

						)",
                            {
                                {"transpose_B", (bool)transpose_B},
                                {"gridDimX", 1},
                                {"gridDimY", m_gridDim.y},
                                {"dataSize", arg0_shape[1]},
                                {"blockSize", 64},
                            });
                    }
                    else if (arg0_shape == nnfusion::Shape({128, 9216}) &&
                             arg1_shape == nnfusion::Shape({9216, 4096}))
                    {
                        m_gridDim = dim3(128, 4, 1);
                        m_blockDim = dim3(16, 16, 1);
                        templ = nnfusion::codegen::get_content_from_templates(
                            "rocm_adapter/fixed_kernels/gemm/matmul_autotvm_NN_128x9216x4096.h.in");
                    }
                    else if (arg0_shape == nnfusion::Shape({128, 4096}) &&
                             arg1_shape == nnfusion::Shape({4096, 4096}))
                    {
                        m_gridDim = dim3(128, 4, 1);
                        m_blockDim = dim3(16, 16, 1);
                        templ = nnfusion::codegen::get_content_from_templates(
                            "rocm_adapter/fixed_kernels/gemm/matmul_autotvm_NN_128x4096x4096.h.in");
                    }
                    else if (arg0_shape == nnfusion::Shape({64, 25088}) &&
                             arg1_shape == nnfusion::Shape({25088, 4096}))
                    {
                        m_gridDim = dim3(128, 2, 1);
                        m_blockDim = dim3(16, 16, 1);
                        templ = nnfusion::codegen::get_content_from_templates(
                            "rocm_adapter/fixed_kernels/gemm/matmul_autotvm_NN_64x25088x4096.h.in");
                    }
                    else if (arg0_shape == nnfusion::Shape({512, 4096}) &&
                             arg1_shape == nnfusion::Shape({4096, 1024}))
                    {
                        m_gridDim = dim3(16, 8, 1);
                        m_blockDim = dim3(16, 16, 1);
                        templ = nnfusion::codegen::get_content_from_templates(
                            "rocm_adapter/fixed_kernels/gemm/matmul_autotvm_NN_512x4096x1024.h.in");
                    }
                    else if (arg0_shape == nnfusion::Shape({512, 1024}) &&
                             arg1_shape == nnfusion::Shape({1024, 4096}))
                    {
                        m_gridDim = dim3(64, 8, 1);
                        m_blockDim = dim3(16, 16, 1);
                        templ = nnfusion::codegen::get_content_from_templates(
                            "rocm_adapter/fixed_kernels/gemm/matmul_autotvm_NN_512x1024x4096.h.in");
                    }
                    else if (arg0_shape == nnfusion::Shape({512, 1024}) &&
                             arg1_shape == nnfusion::Shape({1024, 1024}))
                    {
                        m_gridDim = dim3(16, 8, 1);
                        m_blockDim = dim3(16, 16, 1);
                        templ = nnfusion::codegen::get_content_from_templates(
                            "rocm_adapter/fixed_kernels/gemm/matmul_autotvm_NN_512x1024x1024.h.in");
                    }
                    else
                        return nullptr;

                    // generic_op->validate_and_infer_types();
                    // auto& cfg = generic_op->localOpConfig.getRoot();

                    LanguageUnit_p _lu(new LanguageUnit(get_function_name()));
                    auto& lu = *_lu;
                    lu.block_begin();
                    lu << templ << "\n";
                    lu.block_end();
                    return _lu;
                }

                LanguageUnit_p emit_dependency() override
                {
                    GENERIC_OP_LOGGING();

                    LanguageUnit_p _lu(new LanguageUnit(get_function_name() + "_dep"));
                    _lu->require(header::cuda);
                    return _lu;
                }

                void set_launch_config() override { GENERIC_OP_LOGGING(); }
            };
        } // namespace cuda
    }     // namespace kernels
} // namespace nnfusion

using namespace nnfusion;
using namespace nnfusion::kernels;

REGISTER_KERNEL_EMITTER(
    "Dot",                                                                        // op_name
    Device(ROCM_GPU).TypeConstraint(element::f32).Tag("cuda_kernel").Priority(2), // attrs
    cuda::GemmFixed)                                                              // constructor

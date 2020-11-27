// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "reverse.hpp"
#include "nnfusion/core/operators/generic_op/generic_op.hpp"

using namespace nnfusion;
using namespace nnfusion::kernels;

cuda::Reverse::Reverse(shared_ptr<KernelContext> ctx)
    : BlockCudaEmitter(ctx)
{
    auto reverse = static_pointer_cast<nnfusion::op::Reverse>(ctx->gnode->get_op_ptr());

    arg_shape = ctx->inputs[0]->get_shape();
    arg_rank = arg_shape.size();
    result_shape = ctx->outputs[0]->get_shape();
    reverse_axes = reverse->get_reversed_axes();

    vector<uint32_t> reverse_axes_flag(arg_rank, 0);
    for (auto a : reverse_axes)
    {
        reverse_axes_flag[a] = 1;
    }

    std::stringstream tag;
    tag << arg_rank << "axes_" << join(reverse_axes, "_");
    custom_tag = tag.str();
}

LanguageUnit_p cuda::Reverse::emit_function_body()
{
    LanguageUnit_p _lu(new LanguageUnit(get_function_name()));
    auto& lu = *_lu;

    if (nnfusion::is_scalar(arg_shape))
    {
        lu << "uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;\n";
        lu << "if (tid < 1)\n";
        lu.block_begin();
        {
            lu << "output0[0] = input0[0];\n";
        }
        lu.block_end();
    }
    else
    {
        vector<uint32_t> reverse_axes_flag(arg_rank, 0);
        for (auto a : reverse_axes)
        {
            reverse_axes_flag[a] = 1;
        }

        // function signature:
        // extern "C" __global__ void kernel(m_context->dtypes[0]* input0, m_context->dtypes[0]* input1, m_context->dtypes[2]* output0)
        auto code = nnfusion::op::create_code_from_template(
            R"(
int input_shape[] = {@input_shape@};
int reverse_axes[] = {@reverse_axes@};
uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
if (tid < @nthreads@) {
    uint32_t input_idx = tid;
    uint32_t output_idx = 0;
    uint32_t stride = 1;
    for(uint32_t i = @rank@; i > 0; i--) {
        uint32_t idx = i - 1;
        uint32_t axes_i_in = input_idx % input_shape[idx];
        input_idx /= input_shape[idx];
        uint32_t axes_i_out = reverse_axes[idx] ? input_shape[idx] - axes_i_in - 1 : axes_i_in;
        output_idx += axes_i_out * stride;
        stride *= input_shape[idx];
    }
    output0[output_idx] = input0[tid];
}
        )",
            {{"input_shape", join(arg_shape)},
             {"reverse_axes", join(reverse_axes_flag)},
             {"nthreads", static_cast<uint32_t>(shape_size(arg_shape))},
             {"rank", static_cast<uint32_t>(arg_rank)}});
        lu << code;
    }

    return _lu;
}

void cuda::Reverse::set_launch_config()
{
    uint32_t nthreads = static_cast<uint32_t>(shape_size(result_shape));
    uint32_t block_size_x = 64;
    uint32_t aligned_grid_size_x = align_to_block_size(nthreads, block_size_x);

    m_gridDim = dim3(aligned_grid_size_x, 1, 1);
    m_blockDim = dim3(block_size_x, 1, 1);
}

LanguageUnit_p cuda::Reverse::emit_dependency()
{
    LanguageUnit_p _lu(new LanguageUnit(get_function_name() + "_dep"));
    _lu->require(header::cuda);
    return _lu;
}

REGISTER_KERNEL_EMITTER(
    "Reverse",                                                                    // op_name
    Device(CUDA_GPU).TypeConstraint(element::f32).Tag("cuda_kernel").Priority(2), // attrs
    cuda::Reverse)                                                                // constructor
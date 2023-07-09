// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "slice.hpp"
#include "../cuda_cudnn.hpp"

using namespace nnfusion;
using namespace nnfusion::kernels;

cuda::Slice::Slice(shared_ptr<KernelContext> ctx)
    : BlockCudaEmitter(ctx)
{
    auto slice_op = static_pointer_cast<nnfusion::op::Slice>(ctx->gnode->get_op_ptr());

    input_shape = nnfusion::Shape(ctx->inputs[0]->get_shape());
    output_shape = nnfusion::Shape(ctx->outputs[0]->get_shape());

    input_type = ctx->inputs[0]->get_element_type().c_type_string();
    output_type = ctx->outputs[0]->get_element_type().c_type_string();
    lower_bounds = slice_op->get_lower_bounds();

    input_strides = row_major_strides(input_shape);
    output_strides = row_major_strides(output_shape);
    slice_strides = slice_op->get_strides();

    data_type_size = ctx->outputs[0]->get_element_type().size();

    // determine if this slice can condunt in inplace
    NNFUSION_CHECK(input_shape.size() == output_shape.size());
    size_t outer_size = 1;
    int i = input_shape.size() - 1;
    for (; i >= 0; i--)
    {
        if (input_shape[i] != output_shape[i])
            break;
    }

    for (i = i - 1; i >= 0; i--)
    {
        outer_size *= output_shape[i];
    }

    if (outer_size == 1)
    {
        is_memcpy = true;
        input_offset = 0;
        size_t inner_size = 1;
        for (int i = input_shape.size() - 1; i >= 0; i--)
        {
            input_offset += lower_bounds[i] * inner_size;
            inner_size *= input_shape[i];
        }

        if (!ctx->annotations)
            ctx->annotations = std::make_shared<Annotations>();
        ctx->annotations->add_in_place_oi_pair(oi_pair(0, 0, false, input_offset * data_type_size));
    }

    std::stringstream tag;
    tag << "cuda_slice_" << input_type << "_" << output_type << "_r_" << output_shape.size()
        << "_i_" << join(input_shape, "_") << "_o_" << join(output_shape, "_") << "_lb_"
        << join(lower_bounds, "_") << "_ss_" << join(slice_strides, "_");
    custom_tag = tag.str();
}

bool cuda::Slice::is_eliminative()
{
    if (is_memcpy && m_context->inputs[0]->get_pool() == m_context->outputs[0]->get_pool() &&
        m_context->inputs[0]->get_pool_offset() + input_offset * data_type_size ==
            m_context->outputs[0]->get_pool_offset())
        return true;
    else
        return false;
}

LanguageUnit_p cuda::Slice::emit_function_body()
{
    LanguageUnit_p _lu(new LanguageUnit(get_function_name()));
    auto& lu = *_lu;

    lu << "uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;\n";
    uint32_t nthreads = static_cast<uint32_t>(shape_size(output_shape));
    lu << "if (tid < " << nthreads << ")\n";
    lu.block_begin();
    {
        if (nnfusion::is_scalar(input_shape))
        {
            lu << "output0[0] = input0[0];\n";
        }
        else
        {
            lu << "uint32_t input_strides[] = {" << join(input_strides) << "};\n";
            lu << "uint32_t output_strides[] = {" << join(output_strides) << "};\n";
            lu << "uint32_t lower_bounds[] = {" << join(lower_bounds) << "};\n";
            lu << "uint32_t slice_strides[] = {" << join(slice_strides) << "};\n";
            lu << "uint32_t input_idx = 0;\n";
            lu << "uint32_t output_idx = tid;\n";
            size_t i = 0;
            for (; i < output_shape.size() - 1; i++)
            {
                lu << "input_idx += (((output_idx / output_strides[" << i << "]) * slice_strides["
                   << i << "]) + "
                           "lower_bounds["
                   << i << "]) * input_strides[" << i << "];\n";
                lu << "output_idx %= output_strides[" << i << "];\n";
            }
            lu << "input_idx += (((output_idx / output_strides[" << i << "]) * slice_strides[" << i
               << "]) + "
                  "lower_bounds["
               << i << "]) * input_strides[" << i << "];\n";
            lu << "output0[tid] = input0[input_idx];\n";
        }
    }

    lu.block_end();

    return _lu;
}

void cuda::Slice::set_launch_config()
{
    uint32_t nthreads = static_cast<uint32_t>(shape_size(output_shape));
    // TODO: currently we set it to 64, will add tuning method later
    uint32_t block_size_x = 64;
    uint32_t aligned_grid_size_x = align_to_block_size(nthreads, block_size_x);

    m_gridDim = dim3(aligned_grid_size_x, 1, 1);
    m_blockDim = dim3(block_size_x, 1, 1);
}

LanguageUnit_p cuda::Slice::emit_dependency()
{
    LanguageUnit_p _lu(new LanguageUnit(get_function_name() + "_dep"));

    _lu->require(header::cuda);

    return _lu;
}

REGISTER_KERNEL_EMITTER(
    "Slice",                                                                      // op_name
    Device(CUDA_GPU).TypeConstraint(element::f32).Tag("cuda_kernel").Priority(2), // attrs
    cuda::Slice)                                                                  // constructor

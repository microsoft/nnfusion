// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "pad.hpp"

using namespace nnfusion;
using namespace nnfusion::kernels;

cuda::Pad::Pad(shared_ptr<KernelContext> ctx)
    : BlockCudaEmitter(ctx)
{
    auto pad = static_pointer_cast<nnfusion::op::Pad>(ctx->gnode->get_op_ptr());
    input_shape = nnfusion::Shape(ctx->inputs[0]->get_shape());
    output_shape = nnfusion::Shape(ctx->outputs[0]->get_shape());
    padding_below = nnfusion::Shape(pad->get_padding_below());
    padding_above = nnfusion::Shape(pad->get_padding_above());
    padding_interior = nnfusion::Shape(pad->get_padding_interior());
    input_type = ctx->inputs[0]->get_element_type().c_type_string();
    output_type = ctx->outputs[0]->get_element_type().c_type_string();

    rank = static_cast<uint32_t>(input_shape.size());

    pad_below = nnfusion::NVShape(input_shape.size(), 0);
    pad_interior = nnfusion::NVShape(input_shape.size(), 1);

    int64_t i = padding_below.size() - 1;
    int64_t j = input_shape.size() - 1;
    for (; i >= 0; i--, j--)
    {
        pad_below[j] = padding_below[i];
        pad_interior[j] = padding_interior[i];
    }

    input_strides = row_major_strides(input_shape);
    output_strides = row_major_strides(output_shape);

    std::stringstream tag;
    tag << rank << "pad_i" << join(input_shape, "_") << "pad_o" << join(output_shape, "_") << "_pb"
        << join(padding_below, "_") << "_pi" << join(padding_interior, "_");
    custom_tag = tag.str();
}

LanguageUnit_p cuda::Pad::emit_function_body()
{
    LanguageUnit_p _lu(new LanguageUnit(get_function_name()));
    auto& lu = *_lu;

    // function signature:
    // extern "C" __global__ void kernel(m_context->dtypes[0]* input0, m_context->dtypes[0]* input1, m_context->dtypes[2]* output0)
    //lu.block_begin();
    {
        lu << "uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;\n";
        lu << m_context->dtypes[0] << "* in = input0;\n";
        lu << m_context->dtypes[1] << "* pad = input1;\n";
        lu << m_context->dtypes[2] << "* out = output0;\n";

        uint32_t nthreads = static_cast<uint32_t>(shape_size(output_shape));
        lu << "if (tid < " << nthreads << ")\n";
        lu.block_begin();
        {
            auto expand_vector_size = [](string name, vector<size_t>& d) {
                stringstream ss;
                for (int i = 0; i < d.size(); i++)
                    ss << "size_t " << name << i << " = " << to_string(d[i]) << ";\n";
                return ss.str();
            };

            auto expand_vector_uint32 = [](string name, vector<uint32_t>& d) {
                stringstream ss;
                for (int i = 0; i < d.size(); i++)
                    ss << "uint32_t " << name << i << " = " << to_string(d[i]) << ";\n";
                return ss.str();
            };

            lu << expand_vector_size("input_shape", input_shape);
            lu << expand_vector_uint32("input_strides", input_strides);
            lu << expand_vector_uint32("output_strides", output_strides);
            lu << expand_vector_uint32("padding_below", pad_below);
            lu << expand_vector_uint32("padding_interior", pad_interior);

            lu << "bool in_bounds = true;\n";
            lu << "uint32_t output_pixel = tid;\n";
            lu << "uint32_t input_pixel = 0;\n";
            lu << "int32_t input, input_dil;\n";

            for (size_t i = 0; i < rank; i++)
            {
                if (i != 0)
                {
                    lu << "output_pixel %= output_strides" << i - 1 << ";\n";
                }
                lu << "input_dil = output_pixel / output_strides" << i << " - padding_below" << i
                   << ";\n";

                lu << "input = input_dil / (padding_interior" << i << " + 1);\n";
                lu << "input_dil %= (padding_interior" << i << " + 1);\n";
                lu << "in_bounds = in_bounds && (input >= 0) && (input < input_shape" << i
                   << ") && (input_dil == 0);\n";
                lu << "input_pixel += input * input_strides" << i << ";\n";
            }
            lu << "out[tid] = (in_bounds) ? in[input_pixel] : *pad;\n";
        }
        lu.block_end();
    }
    //lu.block_end();
    return _lu;
}

void cuda::Pad::set_launch_config()
{
    uint32_t nthreads = static_cast<uint32_t>(shape_size(output_shape));
    uint32_t block_size_x = 64;
    uint32_t aligned_grid_size_x = align_to_block_size(nthreads, block_size_x);

    m_gridDim = dim3(aligned_grid_size_x, 1, 1);
    m_blockDim = dim3(block_size_x, 1, 1);
}

LanguageUnit_p cuda::Pad::emit_dependency()
{
    LanguageUnit_p _lu(new LanguageUnit(get_function_name() + "_dep"));
    _lu->require(header::cuda);
    return _lu;
}

bool cuda::Pad::is_eliminative()
{
    if (m_context->inputs[0]->is_same_address(m_context->outputs[0]))
        return true;
    else
        return false;
}

// Register Pad kernel emitter

/*
KernelRegistrar kernel_registrar0(
    "Pad",
    Name("Pad")
        .Device(CUDA_GPU)
        .TypeConstraint(element::f32)
        .Tag("cuda_kernel")
        .KernelFactory([](shared_ptr<KernelContext> context) -> shared_ptr<KernelEmitter> {
            return make_shared<cuda::Pad>(context);
        })
        .Build());
*/

REGISTER_KERNEL_EMITTER(
    "Pad",                                                                        // op_name
    Device(CUDA_GPU).TypeConstraint(element::f32).Tag("cuda_kernel").Priority(2), // attrs
    cuda::Pad)                                                                    // constructor
// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "strided_slice_grad.hpp"
#include "nnfusion/core/operators/generic_op/generic_op.hpp"

using namespace nnfusion;
using namespace nnfusion::kernels;

cuda::StridedSliceGrad::StridedSliceGrad(shared_ptr<KernelContext> ctx)
    : BlockCudaEmitter(ctx)
{
    auto strided_slice_grad =
        static_pointer_cast<nnfusion::op::GenericOp>(ctx->gnode->get_op_ptr());

    x_shape = nnfusion::Shape(ctx->inputs[0]->get_shape());
    begin_shape = nnfusion::Shape(ctx->inputs[1]->get_shape());
    end_shape = nnfusion::Shape(ctx->inputs[2]->get_shape());
    strides_shape = nnfusion::Shape(ctx->inputs[3]->get_shape());
    grad_shape = nnfusion::Shape(ctx->inputs[4]->get_shape());

    x_size = x_shape[0];
    output_shape = nnfusion::Shape(ctx->outputs[0]->get_shape());

    std::stringstream tag;
    tag << join(x_shape, "_") << join(begin_shape, "_") << join(end_shape, "_")
        << join(strides_shape, "_") << join(grad_shape, "_")
        << "Output:" << join(output_shape, "_");
    custom_tag = tag.str();
}

LanguageUnit_p cuda::StridedSliceGrad::emit_function_body()
{
    LanguageUnit_p _lu(new LanguageUnit(get_function_name()));
    auto& lu = *_lu;

    // function signature:
    // extern "C" __global__ void kernel(m_context->dtypes[0]* input0, m_context->dtypes[0]* input1, m_context->dtypes[2]* output0)
    lu << "int* x = input0;\n";
    lu << "int* begin = input1;\n";
    lu << "int* end = input2;\n";
    lu << "int end_fix[10];\n";
    lu << "int* strides = input3;\n";
    lu << "float* grad = input4;\n";
    lu << "float* out = output0;\n";

    lu << "uint32_t nthreads = (end[" << x_size - 1 << "] - begin[" << x_size - 1 << "]) / strides["
       << x_size - 1 << "];\n";
    lu << "if(nthreads == 0)\n"
       << "nthreads = x[" << x_size - 1 << "] / strides[" << x_size - 1 << "];\n";
    lu << "uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;\n";
    lu << "out[i] = 0;\n";
    lu << "if (i < nthreads)\n";
    lu.block_begin();
    {
        lu << "for(int k=0; k < " << x_size << "; k++)\n";
        lu.block_begin();
        {
            lu << "if(begin[k] == end[k]) end_fix[k] = x[k];\n"
               << "else end_fix[k] = end[k];\n";
        }
        lu.block_end();

        //slice_begin_position
        lu << "int begin_pos = 0;\n";
        lu << "for(int k=0; k < " << x_size - 1 << "; k++)\n";
        lu.block_begin();
        {
            lu << "begin_pos += begin[k] * strides[k+1];\n";
        }
        lu.block_end();
        lu << "begin_pos += i * strides[" << x_size - 1 << "];\n";

        // slice_block_stride
        lu << "int block_stride = x[" << x_size - 1 << "] * strides[" << x_size - 2 << "];\n";

        // slice_block_num
        lu << "int block_num = 1;\n";
        lu << "for(int k = 0; k < " << x_size - 1 << "; k++)\n";
        lu.block_begin();
        {
            lu << "block_num *= ((end_fix[k] - begin[k]) / strides[k]);\n";
        }
        lu.block_end();

        // stride_slice_grad
        lu << "for(int k = 0; k < block_num; k++)\n";
        lu.block_begin();
        {
            lu << "out[begin_pos + k * block_stride] = grad[i + k * nthreads];\n";
        }
        lu.block_end();
    }
    lu.block_end();

    return _lu;
}

void cuda::StridedSliceGrad::set_launch_config()
{
    uint32_t nthreads = static_cast<uint32_t>(shape_size(output_shape));
    uint32_t block_size_x = 64;
    uint32_t aligned_grid_size_x = align_to_block_size(nthreads, block_size_x);

    m_gridDim = dim3(aligned_grid_size_x, 1, 1);
    m_blockDim = dim3(block_size_x, 1, 1);
}

LanguageUnit_p cuda::StridedSliceGrad::emit_dependency()
{
    LanguageUnit_p _lu(new LanguageUnit(get_function_name() + "_dep"));
    _lu->require(header::cuda);
    return _lu;
}
REGISTER_KERNEL_EMITTER(
    "StridedSliceGrad",                                                           // op_name
    Device(CUDA_GPU).TypeConstraint(element::f32).Tag("cuda_kernel").Priority(2), // attrs
    cuda::StridedSliceGrad)                                                       // constructor
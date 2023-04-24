// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "scatter_nd.hpp"

using namespace nnfusion;
using namespace nnfusion::kernels;

DECLARE_int32(fmax_block_dim);

cuda::ScatterND::ScatterND(shared_ptr<KernelContext> ctx)
    : BlockCudaEmitter(ctx)
{
    update_shape = static_cast<uint32_t>(shape_size(ctx->inputs[2]->get_shape()));
}

LanguageUnit_p cuda::ScatterND::emit_function_body()
{
    LanguageUnit_p _lu(new LanguageUnit(get_function_name()));
    auto& lu = *_lu;
    lu << "uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;\n";
    lu << "if (tid >= " << update_shape << ") { return; }\n";
    if (shape_size(m_context->inputs[1]->get_shape()) == 1) {
        lu << "output0[" << update_shape <<"* input1[0] + tid] = input2[tid];\n";
    } else {
        std::vector<size_t> stride;
        auto input0_shape = m_context->inputs[0]->get_shape();
        auto input1_shape = m_context->inputs[1]->get_shape();
        auto output0_shape = m_context->outputs[0]->get_shape();
        size_t output0_size = shape_size(output0_shape);
        for (size_t i = 0; i < output0_shape.size(); i++) {
            stride.push_back(output0_size / output0_shape[i]);
            output0_size = output0_size / output0_shape[i];
        }
        size_t K = input1_shape[input1_shape.size() - 1];
        size_t n = shape_size(input1_shape) / K;
        lu << "uint32_t out_id = tid / " << input0_shape[input0_shape.size() - 1] << ";\n";
        lu << "uint32_t in_id = tid % " << input0_shape[input0_shape.size() - 1] << ";\n";
        lu << "uint32_t idx = in_id";
        for (size_t i = 0; i < input1_shape[input1_shape.size() - 1]; i++) {
            lu << " + input1[out_id * " << K << " + " << i << "] * " << stride[i];
        }
        lu << ";\n";
        lu << "output0[idx] = input2[tid];\n";
    }
    return _lu;
}

void cuda::ScatterND::set_launch_config()
{
    uint32_t nthreads = update_shape;
    // TODO: currently we set it to 64, will add tuning method later
    uint32_t block_size_x = FLAGS_fmax_block_dim;
    uint32_t aligned_grid_size_x = align_to_block_size(nthreads, block_size_x);

    m_gridDim = dim3(aligned_grid_size_x, 1, 1);
    m_blockDim = dim3(block_size_x, 1, 1);
}

LanguageUnit_p cuda::ScatterND::emit_dependency()
{
    LanguageUnit_p _lu(new LanguageUnit(get_function_name() + "_dep"));

    _lu->require(header::cuda);

    return _lu;
}

REGISTER_KERNEL_EMITTER(
    "ScatterND",                                                                      // op_name
    Device(CUDA_GPU).TypeConstraint(element::f32).Tag("cuda_kernel").Priority(2), // attrs
    cuda::ScatterND)                                                                  // constructor

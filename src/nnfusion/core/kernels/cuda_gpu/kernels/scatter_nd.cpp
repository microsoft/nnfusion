// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "scatter_nd.hpp"

using namespace nnfusion;
using namespace nnfusion::kernels;

cuda::ScatterND::ScatterND(shared_ptr<KernelContext> ctx)
    : BlockCudaEmitter(ctx)
{
    NNFUSION_CHECK(shape_size(ctx->inputs[1]->get_shape()) == 1);
    update_shape = static_cast<uint32_t>(shape_size(ctx->inputs[2]->get_shape()));
}

LanguageUnit_p cuda::ScatterND::emit_function_body()
{
    LanguageUnit_p _lu(new LanguageUnit(get_function_name()));
    auto& lu = *_lu;
    lu << "uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;\n";
    lu << "if (tid >= " << update_shape << ") { return; }\n";
    lu << "output0[" << update_shape <<"* input1[0] + tid] = input2[tid];\n";
    return _lu;
}

void cuda::ScatterND::set_launch_config()
{
    uint32_t nthreads = update_shape;
    // TODO: currently we set it to 64, will add tuning method later
    uint32_t block_size_x = 128;
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

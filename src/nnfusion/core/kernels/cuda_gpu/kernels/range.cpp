// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "range.hpp"
#include "nnfusion/core/operators/generic_op/generic_op.hpp"

using namespace nnfusion;
using namespace nnfusion::kernels;

cuda::Range::Range(shared_ptr<KernelContext> ctx)
    : BlockCudaEmitter(ctx)
{
    auto range = static_pointer_cast<nnfusion::op::GenericOp>(ctx->gnode->get_op_ptr());
    output_shape = nnfusion::Shape(ctx->outputs[0]->get_shape());

    start = range->localOpConfig.getRoot()["start"];
    limit = range->localOpConfig.getRoot()["limit"];
    delta = range->localOpConfig.getRoot()["delta"];

    range_num = (int)((limit - start + delta - 1) / delta);

    std::stringstream tag;
    tag << "Range "
        << "_Start" << start << "_Limit" << limit << "_Delta" << delta << join(output_shape, "_");
    custom_tag = tag.str();
}

LanguageUnit_p cuda::Range::emit_function_body()
{
    LanguageUnit_p _lu(new LanguageUnit(get_function_name()));
    auto& lu = *_lu;

    // function signature:
    // extern "C" __global__ void kernel(m_context->dtypes[0]* input0, m_context->dtypes[0]* input1, m_context->dtypes[2]* output0)
    lu << m_context->dtypes[0] << "* out = output0;\n";

    uint32_t nthreads = range_num;
    lu << "uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;\n";
    lu << "if (i < " << nthreads << ")\n";
    lu.block_begin();
    {
        lu << "out[i] = " << start << " + " << delta << " * i;\n";
    }
    lu.block_end();

    return _lu;
}

void cuda::Range::set_launch_config()
{
    uint32_t nthreads = static_cast<uint32_t>(shape_size(output_shape));
    uint32_t block_size_x = 64;
    uint32_t aligned_grid_size_x = align_to_block_size(nthreads, block_size_x);

    m_gridDim = dim3(aligned_grid_size_x, 1, 1);
    m_blockDim = dim3(block_size_x, 1, 1);
}

LanguageUnit_p cuda::Range::emit_dependency()
{
    LanguageUnit_p _lu(new LanguageUnit(get_function_name() + "_dep"));
    _lu->require(header::cuda);
    return _lu;
}
REGISTER_KERNEL_EMITTER(
    "Range",                                                                      // op_name
    Device(CUDA_GPU).TypeConstraint(element::f32).Tag("cuda_kernel").Priority(2), // attrs
    cuda::Range)                                                                  // constructor
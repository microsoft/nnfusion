// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "loop.hpp"
#include "../cuda_cudnn.hpp"
#include "convolution.hpp"

using namespace nnfusion;
using namespace nnfusion::kernels;

cuda::Loop::Loop(shared_ptr<KernelContext> ctx)
    : KernelEmitter(ctx)
{
    std::stringstream tag;
    tag << "_LoopOP";
    custom_tag = tag.str();
}

LanguageUnit_p cuda::Loop::emit_function_body()
{
    LanguageUnit_p _lu(new LanguageUnit(get_function_name()));
    auto& lu = *_lu;

    // function signature:
    // extern "C" __global__ void kernel(m_context->dtypes[0]* input0, m_context->dtypes[0]* input1, m_context->dtypes[2]* output0)
    lu << "// TODO\n";
    return _lu;
}

LanguageUnit_p cuda::Loop::emit_dependency()
{
    LanguageUnit_p _lu(new LanguageUnit(get_function_name() + "_dep"));
    _lu->require(header::cuda);
    return _lu;
}

REGISTER_KERNEL_EMITTER("Loop",                                                    // op_name
                        Device(CUDA_GPU).TypeConstraint(element::f32).Priority(2), // attrs
                        cuda::Loop)
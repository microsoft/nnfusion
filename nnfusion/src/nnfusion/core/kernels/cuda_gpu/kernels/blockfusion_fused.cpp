// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "blockfusion_fused.hpp"

using namespace nnfusion;
using namespace nnfusion::kernels;
using namespace nnfusion::kernels::cuda;

BlockFusionFused::BlockFusionFused(shared_ptr<KernelContext> ctx, FunctionUnit_p fn)
    : CudaEmitter(ctx)
{
    set_blockfusion_function(fn);
}

void BlockFusionFused::check_codegen()
{
    NNFUSION_CHECK_NOT_NULLPTR(blockfusion_function)
        << "BlockFusionFused needs FunctionUnit_p from BlockFusionCudaCodegen";
}

void BlockFusionFused::set_blockfusion_function(FunctionUnit_p fn)
{
    if (fn == nullptr)
    {
        NNFUSION_LOG(NNFUSION_WARNING)
            << "BlockFusionFused needs FunctionUnit_p from BlockFusionCudaCodegen. This "
               "warning may happen in KernelContext initialization, which can be ignored.";
    }
    blockfusion_function = fn;
}

LanguageUnit_p BlockFusionFused::emit_function_name()
{
    check_codegen();
    return blockfusion_function->name_unit;
}

LanguageUnit_p BlockFusionFused::emit_function_body()
{
    check_codegen();
    return blockfusion_function->body_unit;
}

LanguageUnit_p BlockFusionFused::emit_function_signature()
{
    check_codegen();
    return blockfusion_function->signature_unit;
}

LanguageUnit_p BlockFusionFused::emit_dependency()
{
    check_codegen();
    return blockfusion_function->dep_unit;
}

LanguageUnit_p BlockFusionFused::emit_function_call()
{
    check_codegen();
    return blockfusion_function->call_unit;
}

LanguageUnit_p BlockFusionFused::emit_comments()
{
    check_codegen();
    return blockfusion_function->comment_unit;
}

void BlockFusionFused::set_launch_config()
{
}

REGISTER_KERNEL_EMITTER(
    "BlockFusionFused",                                                           // op_name
    Device(CUDA_GPU).TypeConstraint(element::f32).Tag("cuda_kernel").Priority(2), // attrs
    cuda::BlockFusionFused)
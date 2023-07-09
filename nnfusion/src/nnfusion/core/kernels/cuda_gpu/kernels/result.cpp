// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "result.hpp"
#include "../cuda_cudnn.hpp"

using namespace nnfusion;
using namespace nnfusion::kernels;

DECLARE_bool(fextern_result_memory);
DECLARE_bool(fhost_entry);

cuda::Result::Result(shared_ptr<KernelContext> ctx)
    : CudaLibEmitter(ctx)
{
    NNFUSION_CHECK(ctx->inputs.size() == 1) << "Input size mismatches.";
    NNFUSION_CHECK(ctx->outputs.size() == 1) << "Output size mismatches.";

    auto result_op = static_pointer_cast<nnfusion::op::Result>(ctx->gnode->get_op_ptr());
    need_copy_to_host = result_op->needs_copy_to_host();
    std::stringstream tag;
    tag << "cuda_result";
    custom_tag = tag.str();
}

LanguageUnit_p cuda::Result::emit_function_signature()
{
    LanguageUnit_p _lu(new LanguageUnit(this->m_kernel_name + "_sig"));
    auto& lu = *_lu;

    vector<string> params;
    params.push_back(m_context->inputs[0]->get_element_type().c_type_string() + "* input0");
    if (need_copy_to_host && !FLAGS_fextern_result_memory && !FLAGS_fhost_entry)
    {
        params.push_back(m_context->outputs[0]->get_element_type().c_type_string() + "** output0");
    }
    else
    {
        params.push_back(m_context->outputs[0]->get_element_type().c_type_string() + "* output0");
    }

    if (FLAGS_fextern_result_memory)
    {
        lu << "void (cudaStream_t stream, " << join(params, ", ") << ")";
    }
    else
    {
        lu << "void (" << join(params, ", ") << ")";
    }

    return _lu;
}

LanguageUnit_p cuda::Result::emit_function_body()
{
    LanguageUnit_p _lu(new LanguageUnit(get_function_name()));
    if (need_copy_to_host)
    {
        if (FLAGS_fextern_result_memory)
        {
            // auto& dst = m_context->outputs[0];
            // auto& src = m_context->inputs[0];
            // *_lu << dst->get_element_type().c_type_string() << "* " << dst->get_name()
            //      << " = output0;\n";
            // *_lu << src->get_element_type().c_type_string() << "* " << src->get_name()
            //      << " = input0;\n";
            // emit_memcpyDtD(*_lu, dst, src);
            // *_lu << "CUDA_SAFE_CALL(cudaMemcpyAsync(output0, input0, 4, cudaMemcpyDeviceToDevice, stream));";
            *_lu << "if (input0 != output0)\n";
            *_lu << "    CUDA_SAFE_CALL(cudaMemcpyAsync(output0, input0,"
                 << m_context->outputs[0]->size() << ", cudaMemcpyDeviceToDevice, stream));";
        }
        else
        {
            *_lu << "*output0 = input0;";
        }
    }
    return _lu;
}

LanguageUnit_p cuda::Result::emit_dependency()
{
    LanguageUnit_p _lu(new LanguageUnit(get_function_name() + "_dep"));

    _lu->require(header::cuda);
    _lu->require(macro::CUDA_SAFE_CALL);

    return _lu;
}

bool cuda::Result::is_eliminative()
{
    if (FLAGS_fhost_entry && m_context->inputs[0]->is_same_address(m_context->outputs[0]))
        return true;
    else
        return false;
}

REGISTER_KERNEL_EMITTER(
    "Result",                                                                  // op_name
    Device(CUDA_GPU).TypeConstraint(element::f32).Tag("cuda_lib").Priority(2), // attrs
    cuda::Result)                                                              // constructor
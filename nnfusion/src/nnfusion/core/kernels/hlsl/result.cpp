// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include <iostream>
#include <stdio.h>

#include "hlsl_kernel_emitter.hpp"
#include "nnfusion/common/languageunit.hpp"
#include "nnfusion/core/kernels/kernel_registration.hpp"
#include "nnfusion/core/operators/generic_op/generic_op.hpp"

DECLARE_string(fhlsl_codegen_type);
DECLARE_bool(fextern_result_memory);
DECLARE_bool(fhost_entry);
namespace nnfusion
{
    namespace kernels
    {
        namespace hlsl
        {
            class Result : public HLSLKernelEmitter
            {
            public:
                Result(shared_ptr<KernelContext> ctx)
                    : HLSLKernelEmitter(ctx)
                {
                    auto result_op =
                        static_pointer_cast<nnfusion::op::Result>(ctx->gnode->get_op_ptr());
                    need_copy_to_host = result_op->needs_copy_to_host();
                }

                LanguageUnit_p emit_function_body() override
                {
                    LanguageUnit_p _lu(new LanguageUnit(get_function_name()));
                    if (FLAGS_fhlsl_codegen_type == "csharp")
                    {
                        *_lu << "output0 = input0;";
                    }
                    else if (FLAGS_fhlsl_codegen_type == "cpp")
                    {
                        if (FLAGS_fextern_result_memory)
                        {
                            *_lu << "if (input0 != output0)\n";
                            *_lu << "    dxMemcpyDtoDAsync(output0, input0, "
                                 << m_context->outputs[0]->size() << ", 0);";
                        }
                        else
                        {
                            *_lu << "*output0 = input0;";
                        }
                    }
                    return _lu;
                }

                LanguageUnit_p emit_function_call() override
                {
                    LanguageUnit_p _lu(new LanguageUnit(this->m_kernel_name + "_call"));
                    auto& lu = *_lu;

                    auto curr = m_context->gnode;

                    if (FLAGS_fhlsl_codegen_type != "default")
                    {
                        vector<string> names;
                        names.insert(names.end(),
                                     m_context->input_names.begin(),
                                     m_context->input_names.end());
                        names.insert(names.end(),
                                     m_context->output_names.begin(),
                                     m_context->output_names.end());
                        names.insert(names.end(),
                                     m_context->tensor_names.begin(),
                                     m_context->tensor_names.end());
                        lu << "(" << join(names, ", ") << ");\n";
                    }
                    else
                    {
                        lu << "NNfusionMemcpy op_" << m_context->output_names[0]
                           << "(device, nullptr, ts_" << m_context->input_names[0] << ");\n\n";
                    }
                    return _lu;
                }

                LanguageUnit_p emit_function_signature() override
                {
                    LanguageUnit_p _lu(new LanguageUnit(this->m_kernel_name + "_sig"));
                    auto& lu = *_lu;

                    vector<string> params;
                    for (size_t i = 0; i < m_context->inputs.size(); i++)
                    {
                        stringstream ss;
                        // ss << m_context->inputs[i]->get_element_type().c_type_string() << "* ";
                        if (FLAGS_fhlsl_codegen_type == "cpp")
                        {
                            ss << "void* input" << i;
                        }
                        else if (FLAGS_fhlsl_codegen_type == "csharp")
                        {
                            ss << "IntPtr input" << i;
                        }
                        params.push_back(ss.str());
                    }

                    for (size_t i = 0; i < m_context->outputs.size(); i++)
                    {
                        stringstream ss;
                        // ss << m_context->outputs[i]->get_element_type().c_type_string() << "* ";
                        if (FLAGS_fhlsl_codegen_type == "cpp")
                        {
                            if (need_copy_to_host && !FLAGS_fextern_result_memory &&
                                !FLAGS_fhost_entry)
                                ss << "void** output" << i;
                            else
                                ss << "void* output" << i;
                        }
                        else if (FLAGS_fhlsl_codegen_type == "csharp")
                        {
                            ss << "IntPtr output" << i;
                        }

                        params.push_back(ss.str());
                    }

                    lu << "void "
                       << "(" << join(params, ", ") << ")";
                    return _lu;
                }

                bool is_eliminative() override
                {
                    if (FLAGS_fhost_entry &&
                        m_context->inputs[0]->is_same_address(m_context->outputs[0]))
                        return true;
                    else
                        return false;
                }

            private:
                bool need_copy_to_host;
            };
        }
    }
}

using namespace nnfusion;
using namespace nnfusion::kernels;

REGISTER_KERNEL_EMITTER("Result",
                        Device(HLSL).TypeConstraint(element::f32).Tag("hlsl_kernel"),
                        hlsl::Result)
// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include <iostream>
#include <stdio.h>

#include "hlsl_kernel_emitter.hpp"
#include "nnfusion/common/languageunit.hpp"
#include "nnfusion/core/kernels/kernel_registration.hpp"
#include "nnfusion/core/operators/generic_op/generic_op.hpp"

DECLARE_string(fhlsl_codegen_type);

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
                }

                LanguageUnit_p emit_function_body() override
                {
                    LanguageUnit_p _lu(new LanguageUnit(get_function_name()));
                    if (FLAGS_fhlsl_codegen_type != "default")
                    {
                        auto& lu = *_lu;
                        lu << "output0 = input0;\n";
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
            };
        }
    }
}

using namespace nnfusion;
using namespace nnfusion::kernels;

REGISTER_KERNEL_EMITTER("Result",
                        Device(HLSL).TypeConstraint(element::f32).Tag("hlsl_kernel"),
                        hlsl::Result)
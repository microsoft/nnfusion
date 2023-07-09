// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include <iostream>
#include <stdio.h>

#include "hlsl_kernel_emitter.hpp"
#include "nnfusion/common/languageunit.hpp"
#include "nnfusion/core/kernels/kernel_registration.hpp"
#include "nnfusion/core/operators/generic_op/generic_op.hpp"

namespace nnfusion
{
    namespace kernels
    {
        namespace hlsl
        {
            class Parameter : public HLSLKernelEmitter
            {
            public:
                Parameter(shared_ptr<KernelContext> ctx)
                    : HLSLKernelEmitter(ctx)
                {
                }

                LanguageUnit_p emit_function_body() override
                {
                    LanguageUnit_p _lu(new LanguageUnit(get_function_name()));
                    return _lu;
                }

                LanguageUnit_p emit_function_call() override
                {
                    LanguageUnit_p _lu(new LanguageUnit(this->m_kernel_name + "_call"));
                    auto& lu = *_lu;

                    auto curr = m_context->gnode;
                    lu << "NNfusionTensor ts_" << m_context->output_names[0] << "(device, {"
                       << nnfusion::codegen::join_collections(
                              curr->get_output_shape(0),
                              [](int idx, ssize_t it) { return std::to_string(it); })
                       << "}, sizeof(" << curr->get_output_element_type(0).c_type_string()
                       << "));\n";

                    lu << "  NNfusionMemcpy op_" << m_context->output_names[0] << "(device, ts_"
                       << m_context->output_names[0] << ", load_data<"
                       << curr->get_output_element_type(0).c_type_string() << ">(\"\", ts_"
                       << m_context->output_names[0] << ".NumElements()));\n\n";
                    return _lu;
                }
            };
        }
    }
}

using namespace nnfusion;
using namespace nnfusion::kernels;
REGISTER_KERNEL_EMITTER("Parameter",
                        Device(HLSL).TypeConstraint(element::f32).Tag("hlsl_kernel"),
                        hlsl::Parameter)
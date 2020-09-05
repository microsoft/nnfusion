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
                    return _lu;
                }

                LanguageUnit_p emit_function_call() override
                {
                    LanguageUnit_p _lu(new LanguageUnit(this->m_kernel_name + "_call"));
                    auto& lu = *_lu;

                    auto curr = m_context->gnode;
                    lu << "NNfusionMemcpy op_" << m_context->output_names[0]
                       << "(device, nullptr, ts_" << m_context->input_names[0] << ");\n\n";
                    return _lu;
                }
            };
        }
    }
}

using namespace nnfusion;
using namespace nnfusion::kernels;

REGISTER_KERNEL_EMITTER("Result",
                        Device(HLSL).TypeConstraint(DT_FLOAT).Tag("hlsl_kernel"),
                        hlsl::Result)
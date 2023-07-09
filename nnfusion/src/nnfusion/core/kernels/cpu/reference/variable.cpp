// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include <iostream>
#include <stdio.h>

#include "nnfusion/common/languageunit.hpp"
#include "nnfusion/core/kernels/kernel_emitter.hpp"
#include "nnfusion/core/kernels/kernel_registration.hpp"
#include "nnfusion/core/operators/generic_op/generic_op.hpp"

namespace nnfusion
{
    namespace kernels
    {
        namespace cpu
        {
            class Variable : public KernelEmitter
            {
            public:
                Variable(shared_ptr<KernelContext> ctx)
                    : KernelEmitter(ctx, "cpu")
                {
                    op = static_pointer_cast<nnfusion::op::Variable>(ctx->gnode->get_op_ptr());
                    NNFUSION_CHECK_NOT_NULLPTR(op) << "Node type is not Variable.";

                    auto element_num =
                        nnfusion::shape_size(ctx->gnode->get_output_tensor(0).get_shape());

                    std::stringstream tag;
                    tag << "variable_" << op->get_name();
                    custom_tag = tag.str();
                }

                LanguageUnit_p emit_function_body() override
                {
                    LanguageUnit_p _lu(new LanguageUnit(get_function_name()));
                    auto& writer = *_lu;
                    auto code = nnfusion::op::create_code_from_template(
                        R"(
                        for (size_t i = 0; i < @element_num@; ++i)
                            output0[i] = 1.0f;
                                        )",
                        {{"element_num", element_num}});
                    writer << code << "\n";

                    return _lu;
                }

                LanguageUnit_p emit_dependency() override

                {
                    LanguageUnit_p _lu(new LanguageUnit(get_function_name() + "_dep"));
                    return _lu;
                }

            private:
                shared_ptr<KernelContext> kernel_ctx;
                shared_ptr<nnfusion::op::Variable> op;
                size_t element_num;
            };
        } // namespace cuda
    }     // namespace kernels
} // namespace nnfusion

// Register Pad kernel emitter

using namespace nnfusion;
using namespace nnfusion::kernels;
REGISTER_KERNEL_EMITTER("Variable",                                       //op_name
                        Device(GENERIC_CPU).TypeConstraint(element::f32), //attrs
                        cpu::Variable)                                    // constructor
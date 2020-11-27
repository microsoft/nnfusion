// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include <iostream>
#include "nnfusion/common/languageunit.hpp"
#include "nnfusion/core/kernels/kernel_emitter.hpp"
#include "nnfusion/core/kernels/kernel_registration.hpp"
#include "nnfusion/core/operators/generic_op/generic_op.hpp"

//Classes
namespace nnfusion
{
    namespace kernels
    {
        namespace cpu
        {
            class AllRef : public KernelEmitter
            {
            public:
                AllRef(shared_ptr<KernelContext> ctx)
                    : KernelEmitter(ctx)
                    , generic_op(
                          static_pointer_cast<nnfusion::op::GenericOp>(ctx->gnode->get_op_ptr()))
                {
                }

                LanguageUnit_p emit_function_body() override
                {
                    const nnfusion::Shape& input_shape_0 = m_context->inputs[0]->get_shape();

                    int axis = generic_op->localOpConfig.getRoot()["axis"];
                    NNFUSION_CHECK(axis == -1);
                    size_t groups = 1LU;
                    for (auto& it : input_shape_0)
                        groups *= it;

                    // TF reduce_all only support bool type
                    NNFUSION_CHECK(m_context->dtypes[0] == "bool");
                    LanguageUnit lu(get_function_name());

                    auto code = nnfusion::op::create_code_from_template(
                        R"(
	bool all = true;
	for (size_t i = 0; i < @groups@; ++i)
		all = all && input0[i];
	output0[0] = all;
                    )",
                        {{"groups", groups}});
                    lu << code << "\n";
                    return std::make_shared<LanguageUnit>(std::move(lu));
                }

                LanguageUnit_p emit_dependency() override
                {
                    LanguageUnit lu(get_function_name() + "_dep");
                    return std::make_shared<LanguageUnit>(std::move(lu));
                }

            private:
                shared_ptr<nnfusion::op::GenericOp> generic_op;
            };

            REGISTER_KERNEL_EMITTER(
                "All",                                                             // op_name
                Device(GENERIC_CPU).TypeConstraint(element::f32).Tag("reference"), // attrs
                AllRef)                                                            // constructor

        } // namespace cpu
    }     // namespace kernels
} // namespace nnfusion

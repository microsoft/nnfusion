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
            class OneHotRef : public KernelEmitter
            {
            public:
                OneHotRef(shared_ptr<KernelContext> ctx)
                    : KernelEmitter(ctx)
                    , generic_op(
                          static_pointer_cast<nnfusion::op::GenericOp>(ctx->gnode->get_op_ptr()))
                {
                }

                LanguageUnit_p emit_function_body() override
                {
                    const nnfusion::Shape& input_shape_0 = m_context->inputs[0]->get_shape();

                    auto& cfg = generic_op->localOpConfig.getRoot();

                    int axis = cfg["axis"].is_null() ? -1 : (int)cfg["axis"];
                    if (axis < 0)
                        axis = input_shape_0.size() - 1;
                    NNFUSION_CHECK(axis == input_shape_0.size() - 1);
                    size_t groups = 1LU;
                    for (int i = 0; i < input_shape_0.size(); ++i)
                        groups *= input_shape_0[i];

                    LanguageUnit lu(get_function_name());
                    auto code = nnfusion::op::create_code_from_template(
                        R"(
	for (int idx = 0; idx < @groups@; ++idx) {
		for (int i = 0; i < @depth@; ++i)
			output0[idx * @depth@ + i] = @off_value@;
		output0[idx * @depth@ + (int)input0[idx]] = @on_value@;
	}
                    )",
                        {
                            {"groups", groups},
                            {"depth", cfg["depth"]},
                            {"off_value", cfg["off_value"]},
                            {"on_value", cfg["on_value"]},
                        });
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
                "OneHot",                                                          // op_name
                Device(GENERIC_CPU).TypeConstraint(element::f32).Tag("reference"), // attrs
                OneHotRef)                                                         // constructor

        } // namespace cpu
    }     // namespace kernels
} // namespace nnfusion

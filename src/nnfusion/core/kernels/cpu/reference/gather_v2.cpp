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
            class GatherV2 : public KernelEmitter
            {
            public:
                GatherV2(shared_ptr<KernelContext> ctx)
                    : KernelEmitter(ctx)
                    , generic_op(
                          static_pointer_cast<nnfusion::op::GenericOp>(ctx->gnode->get_op_ptr()))
                    , gnode(ctx->gnode)
                {
                }

                LanguageUnit_p emit_function_body() override
                {
                    int axis = generic_op->localOpConfig.getRoot()["axis"];
                    auto output0_shape = gnode->get_output_shape(0);
                    auto input0_shape = gnode->get_input_shape(0);
                    auto input1_shape = gnode->get_input_shape(1);
                    // currently only support input0's rank is 1
                    if (input0_shape.size() > 1 || axis != 0)
                    {
                        return nullptr;
                    }
                    auto code = nnfusion::op::create_code_from_template(
                        R"(
                            for (int i = 0; i < @outsize@; i++) {
                                output0[i] = input0[input1[i]];
                            }
                        )",
                        {
                            {"outsize", shape_size(output0_shape)},
                        });

                    LanguageUnit_p _lu(new LanguageUnit(get_function_name()));
                    auto& lu = *_lu;
                    // function signature:
                    // extern "C" __global__ void kernel(m_context->dtypes[0]* input0)
                    lu.block_begin();
                    lu << code << "\n";
                    lu.block_end();
                    return _lu;
                }

                LanguageUnit_p emit_dependency() override
                {
                    LanguageUnit lu(get_function_name() + "_dep");
                    return std::make_shared<LanguageUnit>(std::move(lu));
                }

            private:
                shared_ptr<nnfusion::op::GenericOp> generic_op;
                shared_ptr<nnfusion::graph::GNode> gnode;
            };

            REGISTER_KERNEL_EMITTER(
                "GatherV2",                                                       // op_name
                Device(GENERIC_CPU).TypeConstraint(element::f32).Tag("reference"), // attrs
                GatherV2)                                                      // constructor

        } // namespace cpu
    }     // namespace kernels
} // namespace nnfusion

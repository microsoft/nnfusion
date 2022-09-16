// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include <iostream>
#include "../../cpu_op_emitter.hpp"
#include "../../cuda_langunit.hpp"
#include "nnfusion/core/operators/generic_op/generic_op.hpp"

//Classes
namespace nnfusion
{
    namespace kernels
    {
        namespace cuda_cpu
        {
            class Convolution : public CPUOpEmitter
            {
            public:
                Convolution(shared_ptr<KernelContext> ctx)
                    : CPUOpEmitter(ctx)
                    , m_node(ctx->gnode)
                {
                }

                LanguageUnit_p emit_function_body() override
                {
                    const nnfusion::Shape& input_shape_0 = m_context->inputs[0]->get_shape();
                    const nnfusion::Shape& input_shape_1 = m_context->inputs[1]->get_shape();

                    NNFUSION_CHECK(input_shape_0.size() == 4);
                    NNFUSION_CHECK(input_shape_1.size() == 4);

                    auto conv_op = dynamic_pointer_cast<op::Convolution>(m_node->get_op_ptr());

                    NNFUSION_LOG(INFO) << "conv_layout: " << conv_op->get_data_format();

                    NNFUSION_CHECK(input_shape_0[2] == 1) << "not implemented";
                    NNFUSION_CHECK(input_shape_0[3] == 1) << "not implemented";
                    NNFUSION_CHECK(input_shape_1[0] == 1) << "not implemented";
                    NNFUSION_CHECK(input_shape_1[2] == 1) << "not implemented";
                    NNFUSION_CHECK(input_shape_1[3] == 1) << "not implemented";
                    NNFUSION_CHECK(conv_op->get_data_format() == "NCHW2CNHW") << "not implemented";
                    LanguageUnit lu(get_function_name());
                    
                    auto code = nnfusion::op::create_code_from_template(
                        R"(
	for (long STEP = 0; STEP < @batch@; ++STEP) {
        @T@ sum = 0;
        for (int r = 0; r < @rsize@; r++) {
            sum += input0[STEP * @rsize@ + r] * input1[r];
        }
        output0[STEP] = sum;
	}
                    )",
                        {
                            {"batch", input_shape_0[0]},
                            {"T", m_context->dtypes[0]}, 
                            {"rsize", input_shape_0[1]}
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
                shared_ptr<nnfusion::graph::GNode> m_node;
            };

            REGISTER_KERNEL_EMITTER(
                "Convolution",                                                     // op_name
                Device(SINGLE_CPU).TypeConstraint(element::f32).Tag("reference"), // attrs
                Convolution)                                                    // constructor

        } // namespace cpu
    }     // namespace kernels
} // namespace nnfusion

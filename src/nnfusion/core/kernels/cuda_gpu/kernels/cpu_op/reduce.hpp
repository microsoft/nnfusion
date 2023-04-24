// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#pragma once
#include "../../cpu_op_emitter.hpp"
#include "../../cuda_langunit.hpp"
#include "nnfusion/core/operators/generic_op/generic_op.hpp"

DECLARE_string(fdefault_device);

namespace nnfusion
{
    namespace kernels
    {
        namespace cuda_cpu
        {
            template <class T>
            class Reduce : public CPUOpEmitter
            {
            public:
                Reduce(shared_ptr<KernelContext> ctx)
                    : CPUOpEmitter(ctx)
                {
                    if (auto reduce =
                            dynamic_pointer_cast<nnfusion::op::Reduce>(ctx->gnode->get_op_ptr()))
                    {
                        reduce_axis = reduce->get_reduction_axes();
                        init_value = " = 0.0";
                        reduce_operator = "r + v";
                    }
                    else if (auto reduce =
                                 dynamic_pointer_cast<nnfusion::op::Max>(ctx->gnode->get_op_ptr()))
                    {
                        reduce_axis = reduce->get_reduction_axes();
                        init_value = " = input0[in_idx]";
                        reduce_operator = "max(r, v)";
                    }
                    else if (auto reduce =
                                 dynamic_pointer_cast<nnfusion::op::Min>(ctx->gnode->get_op_ptr()))
                    {
                        reduce_axis = reduce->get_reduction_axes();
                        init_value = " = input0[in_idx]";
                        reduce_operator = "min(r, v)";
                    }
                    else if (auto reduce = dynamic_pointer_cast<nnfusion::op::Product>(
                                 ctx->gnode->get_op_ptr()))
                    {
                        reduce_axis = reduce->get_reduction_axes();
                        init_value = " = 1.0";
                        reduce_operator = "r * v";
                    }
                    else if (auto reduce =
                                 dynamic_pointer_cast<nnfusion::op::Sum>(ctx->gnode->get_op_ptr()))
                    {
                        reduce_axis = reduce->get_reduction_axes();
                        init_value = " = 0.0";
                        reduce_operator = "r + v";
                    }
                    else if (auto reduce = dynamic_pointer_cast<nnfusion::op::ReduceAny>(
                                 ctx->gnode->get_op_ptr()))
                    {
                        reduce_axis = reduce->get_reduction_axes();
                        init_value = " = false";
                        reduce_operator = "r || v";
                    }
                    else
                    {
                        NNFUSION_CHECK_FAIL() << "incorrect kernel for reduce";
                    }

                    input_shape = nnfusion::Shape(ctx->inputs[0]->get_shape());
                    output_shape = nnfusion::Shape(ctx->outputs[0]->get_shape());
                    input_type = ctx->inputs[0]->get_element_type().c_type_string();
                    output_type = ctx->outputs[0]->get_element_type().c_type_string();
                }

                LanguageUnit_p emit_function_body() override
                {
                    LanguageUnit_p _lu(new LanguageUnit(get_function_name()));
                    auto& lu = *_lu;
                    if (shape_size(output_shape) > 1)
                    {
                        return nullptr;
                    }
                    lu << output_type << " r " << init_value << ";\n";
                    lu << "for (int i = 0; i < " << shape_size(input_shape) << "; i++)";
                    lu.block_begin();
                    lu << output_type << " v = input0[i];\n";
                    lu << "r = " << reduce_operator << ";\n";
                    lu.block_end();
                    lu << "output0[0] = r;";
                    return _lu;
                }

                LanguageUnit_p emit_dependency() override
                {
                    LanguageUnit_p _lu(new LanguageUnit(get_function_name() + "_dep"));
                    return _lu;
                }

            private:
                shared_ptr<KernelContext> kernel_ctx;
                nnfusion::AxisSet reduce_axis;
                nnfusion::Shape input_shape, output_shape;
                string reduce_op, input_type, output_type, init_value, reduce_operator;
            };
        } // namespace cuda
    }     // namespace kernels
} // namespace nnfusion

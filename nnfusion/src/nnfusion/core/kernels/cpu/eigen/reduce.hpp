// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#pragma once
#include "../cpu_kernel_emitter.hpp"
#include "../cpu_kernelops.hpp"
#include "nnfusion/core/operators/generic_op/generic_op.hpp"

namespace nnfusion
{
    namespace kernels
    {
        namespace cpu
        {
            template <typename T>
            class ReduceEigen : public EigenKernelEmitter
            {
            public:
                ReduceEigen(shared_ptr<KernelContext> ctx)
                    : EigenKernelEmitter(ctx)
                {
                    auto op = static_pointer_cast<T>(ctx->gnode->get_op_ptr());
                    NNFUSION_CHECK_NOT_NULLPTR(op) << "Node type is not expected.";

                    reduce_axis = op->get_reduction_axes();
                    input_shape = ctx->inputs[0]->get_shape();
                    output_shape = ctx->outputs[0]->get_shape();
                    data_type = ctx->outputs[0]->get_element_type().c_type_string();
                }

                LanguageUnit_p emit_function_body() override
                {
                    LanguageUnit_p _lu(new LanguageUnit(get_function_name()));
                    auto& lu = *_lu;

                    // Handle the cases that input tensor is matrix.
                    if (CpuOpMap<T>::eigen_op != nullptr)
                    {
                        std::string op = CpuOpMap<T>::eigen_op;
                        auto code = nnfusion::op::create_code_from_template(
                            R"(
Eigen::array<Eigen::Index, @in_rank@> in_dims({@in_dims@});
Eigen::array<Eigen::Index, @out_rank@> out_dims({@out_dims@});
Eigen::array<Eigen::Index, @axis_count@> axes({@axes@});

Eigen::TensorMap<Eigen::Tensor<@ElementType@, @out_rank@, Eigen::RowMajor>> out(
    static_cast<@ElementType@ *>(output0), out_dims);
Eigen::TensorMap<Eigen::Tensor<@ElementType@, @in_rank@, Eigen::RowMajor>> in(
    static_cast<@ElementType@ *>(input0), in_dims);
out.device(*(thread_pool->GetDevice())) = in.@op@(axes);
)",
                            {{"in_rank", input_shape.size()},
                             {"out_rank", output_shape.size()},
                             {"in_dims", join(input_shape)},
                             {"out_dims", join(output_shape)},
                             {"axis_count", reduce_axis.size()},
                             {"axes", join(reduce_axis)},
                             {"ElementType", data_type},
                             {"op", op}});
                        lu << code;
                    }
                    else
                    {
                        return nullptr;
                    }

                    return _lu;
                }

                LanguageUnit_p emit_dependency() override
                {
                    LanguageUnit_p _lu(new LanguageUnit(get_function_name() + "_dep"));
                    _lu->require(header::eigen_tensor);
                    return _lu;
                }

            private:
                string data_type;
                nnfusion::Shape input_shape, output_shape;
                nnfusion::AxisSet reduce_axis;
            };
        } // namespace cpu
    }     // namespace kernels
} // namespace nnfusion

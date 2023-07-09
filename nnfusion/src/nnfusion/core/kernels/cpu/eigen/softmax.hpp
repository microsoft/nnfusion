// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#pragma once
#include "../cpu_kernel_emitter.hpp"
#include "nnfusion/core/operators/generic_op/generic_op.hpp"

namespace nnfusion
{
    namespace kernels
    {
        namespace cpu
        {
            template <typename ElementType>
            class SoftmaxEigen : public EigenKernelEmitter
            {
            public:
                SoftmaxEigen(shared_ptr<KernelContext> ctx)
                    : EigenKernelEmitter(ctx)
                {
                    auto pad = static_pointer_cast<nnfusion::op::Softmax>(ctx->gnode->get_op_ptr());
                    input_shape = nnfusion::Shape(ctx->inputs[0]->get_shape());
                    output_shape = nnfusion::Shape(ctx->outputs[0]->get_shape());
                    axes = pad->get_axes();
                    output_type = ctx->outputs[0]->get_element_type().c_type_string();

                    rank = static_cast<uint32_t>(input_shape.size());

                    std::stringstream tag;
                    tag << rank << "softmax_i" << join(input_shape, "_") << "softmax_o"
                        << join(output_shape, "_") << "_axes" << join(axes, "_");
                    custom_tag = tag.str();
                }

                LanguageUnit_p emit_function_body() override
                {
                    LanguageUnit_p _lu(new LanguageUnit(get_function_name()));
                    auto& lu = *_lu;

                    if (m_context->inputs[0]->get_element_type() == element::f32)
                    {
                        nnfusion::Shape rdims(rank);
                        nnfusion::Shape bcast(rank);
                        for (size_t i = 0; i < rank; ++i)
                        {
                            if (axes.count(i))
                            {
                                rdims[i] = 1;
                            }
                            else
                            {
                                rdims[i] = input_shape[i];
                            }
                        }
                        for (size_t i = 0; i < rank; ++i)
                        {
                            bcast[i] = input_shape[i] / rdims[i];
                        }

                        auto code = nnfusion::op::create_code_from_template(
                            R"(
Eigen::array<Eigen::Index, @Rank@> in_dims({@in_dims@});
Eigen::array<Eigen::Index, @Rank@> rdims({@rdims@});
Eigen::array<Eigen::Index, @Rank@> bcast({@bcast@});
Eigen::array<Eigen::Index, @AxisCount@> axes({@axes@});

Eigen::TensorMap<Eigen::Tensor<@ElementType@, @Rank@, Eigen::RowMajor>> out(
    static_cast<@ElementType@ *>(output0), in_dims),
    in(static_cast<@ElementType@ *>(input0), in_dims);

out.device(*(thread_pool->GetDevice())) =
    (in - in.maximum(axes).eval().reshape(rdims).broadcast(bcast)).exp();
out.device(*(thread_pool->GetDevice())) =
    out * out.sum(axes).inverse().eval().reshape(rdims).broadcast(bcast);
)",
                            {{"Rank", rank},
                             {"AxisCount", axes.size()},
                             {"in_dims", join(input_shape)},
                             {"rdims", join(rdims)},
                             {"bcast", join(bcast)},
                             {"axes", join(axes)},
                             {"ElementType", output_type}});
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
                    _lu->require(header::thread);
                    _lu->require(header::eigen_tensor);

                    return _lu;
                }

            private:
                shared_ptr<KernelContext> kernel_ctx;
                nnfusion::Shape input_shape, output_shape;
                nnfusion::AxisSet axes;
                uint32_t rank;
                string output_type;
            };
        } // namespace cpu
    }     // namespace kernels
} // namespace nnfusion

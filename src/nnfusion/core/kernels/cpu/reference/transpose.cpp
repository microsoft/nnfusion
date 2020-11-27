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
            class TransposeRef : public KernelEmitter
            {
            public:
                TransposeRef(shared_ptr<KernelContext> ctx)
                    : KernelEmitter(ctx)
                    , generic_op(
                          static_pointer_cast<nnfusion::op::GenericOp>(ctx->gnode->get_op_ptr()))
                {
                }

                LanguageUnit_p emit_function_body() override
                {
                    const nnfusion::Shape& input_shape_0 = m_context->inputs[0]->get_shape();

                    std::vector<int> axes_order = generic_op->localOpConfig.getRoot()["axes_order"];
                    NNFUSION_CHECK(axes_order.size() == input_shape_0.size());
                    size_t mul_val = 0, mul_inp = 0;
                    for (int i = 0; i < axes_order.size(); ++i)
                    {
                        NNFUSION_CHECK(axes_order[i] >= 0 && axes_order[i] < axes_order.size());
                        mul_val *= input_shape_0[axes_order[i]];
                        mul_inp *= input_shape_0[i];
                    }
                    NNFUSION_CHECK(mul_val == mul_inp);

                    NNFUSION_CHECK(axes_order.size() <= 4);
                    std::vector<int> st_in(1, 1), st_out(4, -1), reorder(axes_order.size());
                    for (int i = 0; i < axes_order.size(); ++i)
                        reorder[axes_order[i]] = i;
                    std::reverse(reorder.begin(), reorder.end());
                    int last = 1;
                    for (int i = 0; i < reorder.size(); ++i)
                    {
                        if (i > 0)
                            st_in.push_back(st_in.back() * input_shape_0[reorder.size() - i]);
                        st_out[reorder[i]] = last;
                        last *= input_shape_0[reorder[i]];
                    }
                    for (auto& it : st_out)
                        if (it < 0)
                            it = last;
                    while (st_in.size() < 4)
                        st_in.push_back(st_in.back());
                    std::reverse(st_in.begin(), st_in.end());

                    std::vector<int> input_4d(4 - input_shape_0.size(), 1);
                    for (auto& it : input_shape_0)
                        input_4d.push_back(it);

                    auto code = nnfusion::op::create_code_from_template(
                        R"(
        int offset = 0, top = @D_0@ * @D_1@ * @D_2@ * @D_3@;
        while (offset < top) {
                int id_3 = offset % @D_3@;
                int id_2 = offset / @D_3@ % @D_2@;
                int id_1 = offset / @D_3@ / @D_2@ % @D_1@;
                int id_0 = offset / @D_3@ / @D_2@ / @D_1@;

                output0[id_0 * @ST_OUT_0@ + id_1 * @ST_OUT_1@ + id_2 * @ST_OUT_2@ + id_3 * @ST_OUT_3@] =
                        input0[id_0 * @ST_IN_0@ + id_1 * @ST_IN_1@ + id_2 * @ST_IN_2@ + id_3 * @ST_IN_3@];
                offset ++;
        }
)",
                        {
                            {"D_0", input_4d[0]},
                            {"D_1", input_4d[1]},
                            {"D_2", input_4d[2]},
                            {"D_3", input_4d[3]},
                            {"ST_IN_0", st_in[0]},
                            {"ST_IN_1", st_in[1]},
                            {"ST_IN_2", st_in[2]},
                            {"ST_IN_3", st_in[3]},
                            {"ST_OUT_0", st_out[0]},
                            {"ST_OUT_1", st_out[1]},
                            {"ST_OUT_2", st_out[2]},
                            {"ST_OUT_3", st_out[3]},
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
            };

            REGISTER_KERNEL_EMITTER(
                "Transpose",                                                       // op_name
                Device(GENERIC_CPU).TypeConstraint(element::f32).Tag("reference"), // attrs
                TransposeRef)                                                      // constructor

        } // namespace cpu
    }     // namespace kernels
} // namespace nnfusion

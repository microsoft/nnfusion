// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

// This is the 2rd-generation of kernel definition, recommend to extend new ops with this style
// Changes needed for creating an new kernel with 2rd generation style.
//

#include "../cuda_emitter.hpp"
#include "../cuda_langunit.hpp"
#include "nnfusion/core/operators/generic_op/generic_op.hpp"

namespace nnfusion
{
    namespace kernels
    {
        namespace cuda
        {
            class FusedDivAddSoftmax : public CudaLibEmitter
            {
                shared_ptr<nnfusion::op::GenericOp> generic_op;
                nnfusion::Shape input_shape, output_shape;
                int N, D;
                element::Type dtype;

            public:
                FusedDivAddSoftmax(shared_ptr<KernelContext> ctx)
                    : CudaLibEmitter(ctx)
                    , generic_op(
                          static_pointer_cast<nnfusion::op::GenericOp>(ctx->gnode->get_op_ptr()))
                {
                    input_shape = nnfusion::Shape(ctx->inputs[0]->get_shape());
                    output_shape = nnfusion::Shape(ctx->outputs[0]->get_shape());
                    dtype = m_context->inputs[0]->get_element_type();
                    size_t axis = output_shape.size() - 1;

                    N = 1;
                    D = 1;
                    for (size_t i = 0; i < input_shape.size(); i++)
                    {
                        if (i < axis)
                        {
                            N *= input_shape[i];
                        }
                        else
                        {
                            D *= input_shape[i];
                        }
                    }
                }

                LanguageUnit_p emit_function_body() override
                {
                   LanguageUnit_p _lu(new LanguageUnit(get_function_name()));
                   auto& lu = *_lu;
                   auto code = nnfusion::op::create_code_from_template(
                   R"(
                        DispatchSoftmax<@dtype@>(stream, @N@, @D@, input0, output0);
                    )",
                    {{"dtype", (dtype == element::f16) ? "half" : "float"}, {"D", D}, {"N", N}});

                    lu << code << "\n";
                    return _lu;
                }

                LanguageUnit_p emit_dependency() override
                {
                    GENERIC_OP_LOGGING();

                    LanguageUnit_p _lu(new LanguageUnit(get_function_name() + "_dep"));
                    _lu->require(declaration::oneflow_softmax);
                    declaration::oneflow_softmax->require(header::math_constants);
                    declaration::oneflow_softmax->require(header::cub);

                    return _lu;
                }
            };
        } // namespace cuda
    }     // namespace kernels
} // namespace nnfusion

using namespace nnfusion;
using namespace nnfusion::kernels;

REGISTER_KERNEL_EMITTER(
    "FusedDivAddSoftmax",                                                                     // op_name
    Device(CUDA_GPU).TypeConstraint(element::f32).Tag("cuda_kernel").Priority(2), // attrs
    cuda::FusedDivAddSoftmax)                                                                 // constructor

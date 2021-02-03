// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

// #include "../cuda_cudnn.hpp"
#include "../cuda_cudnn.hpp"
#include "../cuda_emitter.hpp"
#include "../cuda_langunit.hpp"
#include "nnfusion/core/operators/generic_op/generic_op.hpp"

namespace nnfusion
{
    namespace kernels
    {
        namespace cuda
        {
            class LayerNorm : public CudaLibEmitter
            {
                shared_ptr<nnfusion::op::GenericOp> generic_op;

            public:
                LayerNorm(shared_ptr<KernelContext> ctx)
                    : CudaLibEmitter(ctx)
                    , generic_op(
                          static_pointer_cast<nnfusion::op::GenericOp>(ctx->gnode->get_op_ptr()))
                {
                    GENERIC_OP_LOGGING();
                }

                LanguageUnit_p emit_function_body() override
                {
                    GENERIC_OP_LOGGING();

                    const nnfusion::Shape& input_shape = m_context->inputs[0]->get_shape();

                    auto& cfg = generic_op->localOpConfig.getRoot();
                    float eps = cfg["epsilon"];
                    int axis = cfg["axis"];
                    axis += axis < 0 ? input_shape.size() : 0;
                    size_t n1 = 1, n2 = 1;
                    for (auto i = 0; i < axis; i++)
                    {
                        n1 *= input_shape[i];
                    }
                    for (auto i = axis; i < input_shape.size(); i++)
                    {
                        n2 *= input_shape[i];
                    }

                    nnfusion::element::Type dtype = m_context->inputs[0]->get_element_type();
                    LanguageUnit_p _lu(new LanguageUnit(get_function_name()));
                    auto& lu = *_lu;
                    // lu << "HostApplyLayerNorm(output0,"
                    //    << " output1,"
                    //    << " output2,"
                    //    << " input0," << n1 << "," << n2 << "," << eps << ","
                    //    << " input1,"
                    //    << " input2);\n";

                    auto code = nnfusion::op::create_code_from_template(
                        R"(
HostApplyLayerNorm<@dtype@>(
output0, output1, output2, input0, @n1@, @n2@, @expression1@@eps@@expression2@, input1, input2);
    )",
                        {{"n1", n1},
                         {"n2", n2},
                         {"dtype", (dtype == element::f16) ? "half" : "float"},
                         {"expression1", (dtype == element::f16) ? "__float2half_rn(" : ""},
                         {"expression2", (dtype == element::f16) ? ")" : ""},
                         {"eps", eps}});

                    lu << code << "\n";
                    return _lu;
                }

                LanguageUnit_p emit_dependency() override
                {
                    GENERIC_OP_LOGGING();

                    LanguageUnit_p _lu(new LanguageUnit(get_function_name() + "_dep"));
                    _lu->require(header::cuda);
                    _lu->require(declaration::cuda_layer_norm);
                    declaration::cuda_layer_norm->require(declaration::math_Rsqrt);
                    declaration::cuda_layer_norm->require(declaration::warp);
                    return _lu;
                }
            };

        } // namespace cuda
    }     // namespace kernels
} // namespace nnfusion

// Register kernel emitter

using namespace nnfusion;
using namespace nnfusion::kernels;

REGISTER_KERNEL_EMITTER("LayerNorm",                                                  // op_name
                        Device(CUDA_GPU).TypeConstraint(element::f32).Tag("cudalib"), // attrs
                        cuda::LayerNorm)                                              // constructor

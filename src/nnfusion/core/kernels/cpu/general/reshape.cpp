// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "../cpu_kernel_emitter.hpp"

namespace nnfusion
{
    namespace kernels
    {
        namespace cpu
        {
            class ReshapeMemcpy : public CpuKernelEmitter
            {
            public:
                ReshapeMemcpy(shared_ptr<KernelContext> ctx)
                    : CpuKernelEmitter(ctx)
                    , is_memcpy(false)
                    , is_noop(false)
                {
                    NNFUSION_CHECK(ctx->outputs[0]->size(false) > 0)
                        << "Invalid output shape for Reshape.";
                    auto reshape =
                        static_pointer_cast<nnfusion::op::Reshape>(ctx->gnode->get_op_ptr());
                    // Noop
                    if (ctx->outputs[0]->get_name() == ctx->inputs[0]->get_name())
                    {
                        is_noop = true;
                        return;
                    }

                    input_shape = ctx->inputs[0]->get_shape();
                    output_shape = ctx->outputs[0]->get_shape();
                    size_t output_size = shape_size(output_shape);

                    // for a zero-size tensor, or change from 1^m shape to 1^n shape, just do a copy
                    if (!reshape->get_is_layout_change() || output_size < 2)
                    {
                        is_memcpy = true;
                        // add inplace tag
                        if (!ctx->annotations)
                            ctx->annotations = std::make_shared<Annotations>();
                        ctx->annotations->add_in_place_oi_pair({0, 0, false});
                    }
                }

                virtual LanguageUnit_p emit_function_body()
                {
                    if (!is_memcpy && !is_noop)
                    {
                        return nullptr;
                    }

                    LanguageUnit_p _lu(new LanguageUnit(get_function_name()));
                    auto& lu = *_lu;

                    if (is_memcpy)
                    {
                        lu << "if (input0 != output0) {\n"
                           << "   memcpy(output0, input0, "
                           << static_cast<uint32_t>(shape_size(input_shape)) << " * sizeof("
                           << m_context->dtypes[0] << "));\n"
                           << "}\n";
                    }
                    else
                    {
                        lu << "// noop as input0 == output0.\n";
                    }

                    return _lu;
                }

                virtual LanguageUnit_p emit_dependency()
                {
                    LanguageUnit_p _lu(new LanguageUnit(get_function_name() + "_dep"));
                    return _lu;
                }

                bool is_eliminative()
                {
                    if (is_memcpy && m_context->inputs[0]->is_same_address(m_context->outputs[0]))
                        return true;
                    else
                        return false;
                }

            private:
                bool is_memcpy = false;
                bool is_noop;
                nnfusion::Shape input_shape, output_shape;
            };
        } // namespace cpu
    }     // namespace kernels
} // namespace nnfusion

using namespace nnfusion;
using namespace nnfusion::kernels;

REGISTER_KERNEL_EMITTER(
    "Reshape",                                                               //op_name
    Device(GENERIC_CPU).TypeConstraint(element::f32).Tag("cpu").Priority(2), //attrs
    cpu::ReshapeMemcpy)                                                      //constructor

// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "../cuda_emitter.hpp"
#include "../cuda_langunit.hpp"
#include "nnfusion/core/operators/generic_op/generic_op.hpp"

namespace nnfusion
{
    namespace kernels
    {
        namespace cuda
        {
            class Zeros : public CudaLibEmitter
            {
                shared_ptr<nnfusion::op::GenericOp> generic_op;

            public:
                Zeros(shared_ptr<KernelContext> ctx)
                    : CudaLibEmitter(ctx)
                    , generic_op(
                          static_pointer_cast<nnfusion::op::GenericOp>(ctx->gnode->get_op_ptr()))
                {
                    GENERIC_OP_LOGGING();
                }

                LanguageUnit_p emit_function_body() override
                {
                    GENERIC_OP_LOGGING();

                    auto size_inbyte = nnfusion::shape_size(m_context->outputs[0]->get_shape()) *
                                       m_context->outputs[0]->get_element_type().bitwidth() / 8;

                    LanguageUnit_p _lu(new LanguageUnit(get_function_name()));
                    auto& lu = *_lu;
                    lu << "CUDA_SAFE_CALL(cudaMemset(output0, 0, " << size_inbyte << "));\n";
                    return _lu;
                }

                LanguageUnit_p emit_dependency() override
                {
                    GENERIC_OP_LOGGING();

                    LanguageUnit_p _lu(new LanguageUnit(get_function_name() + "_dep"));
                    _lu->require(header::cuda);
                    _lu->require(header::stdexcept);
                    _lu->require(header::sstream);
                    _lu->require(macro::CUDA_SAFE_CALL);
                    return _lu;
                }
            };
        } // namespace cuda
    }     // namespace kernels
} // namespace nnfusion

// Register kernel emitter

using namespace nnfusion;
using namespace nnfusion::kernels;

REGISTER_KERNEL_EMITTER("Zeros",                                                          // op_name
                        Device(CUDA_GPU).TypeConstraint(element::f32).Tag("cuda_kernel"), // attrs
                        cuda::Zeros) // constructor

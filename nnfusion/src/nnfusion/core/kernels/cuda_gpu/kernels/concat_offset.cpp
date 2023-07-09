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
            class ConcatOffset : public BlockCudaEmitter
            {
                shared_ptr<nnfusion::op::GenericOp> generic_op;

            public:
                ConcatOffset(shared_ptr<KernelContext> ctx)
                    : BlockCudaEmitter(ctx)
                {
                    GENERIC_OP_LOGGING();
                }

                LanguageUnit_p emit_function_body() override
                {
                    GENERIC_OP_LOGGING();

                    const nnfusion::Shape& input_shape_0 = m_context->inputs[0]->get_shape();

                    auto code = nnfusion::op::create_code_from_template(
                        R"(
// Left blank by purpose.
)",
                        {});

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
                    GENERIC_OP_LOGGING();

                    LanguageUnit_p _lu(new LanguageUnit(get_function_name() + "_dep"));
                    _lu->require(header::cuda);
                    return _lu;
                }

                void set_launch_config() override
                {
                    GENERIC_OP_LOGGING();

                    m_gridDim = dim3(1, 1, 1);
                    m_blockDim = dim3(1, 1, 1);
                }
            };
        } // namespace cuda
    }     // namespace kernels
} // namespace nnfusion

using namespace nnfusion;
using namespace nnfusion::kernels;

REGISTER_KERNEL_EMITTER(
    "ConcatOffset",                                                               // op_name
    Device(CUDA_GPU).TypeConstraint(element::f32).Tag("cuda_kernel").Priority(2), // attrs
    cuda::ConcatOffset)                                                           // constructor

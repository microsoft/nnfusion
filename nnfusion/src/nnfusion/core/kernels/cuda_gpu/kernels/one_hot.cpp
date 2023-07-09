// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

// This is the 2rd-generation of kernel definition, recommend to extend new ops with this style
// Changes needed for creating an new kernel with 2rd generation style.
//

#include "../cuda_emitter.hpp"
#include "../cuda_langunit.hpp"
#include "nnfusion/core/operators/generic_op/generic_op.hpp"

/*********************************

REGISTER_OP(OneHot)
    .attr<int>("axis", -1)
    .attr<int>("depth")
    .attr<nnfusion::op::OpConfig::any>("off_value", 1.0f)
    .attr<nnfusion::op::OpConfig::any>("on_value", 0.0f)
    ...

*********************************/

namespace nnfusion
{
    namespace kernels
    {
        namespace cuda
        {
            class OneHot : public BlockCudaEmitter
            {
                shared_ptr<nnfusion::op::GenericOp> generic_op;
                size_t groups;

            public:
                OneHot(shared_ptr<KernelContext> ctx)
                    : BlockCudaEmitter(ctx)
                    , generic_op(
                          static_pointer_cast<nnfusion::op::GenericOp>(ctx->gnode->get_op_ptr()))
                    , groups(1LU)
                {
                    GENERIC_OP_LOGGING();

                    const nnfusion::Shape& input_shape_0 = m_context->inputs[0]->get_shape();
                    for (int i = 0; i < input_shape_0.size(); ++i)
                        groups *= input_shape_0[i];
                }

                LanguageUnit_p emit_function_body() override
                {
                    GENERIC_OP_LOGGING();

                    const nnfusion::Shape& input_shape_0 = m_context->inputs[0]->get_shape();

                    auto& cfg = generic_op->localOpConfig.getRoot();

                    int axis = cfg["axis"].is_null() ? -1 : (int)cfg["axis"];
                    if (axis < 0)
                        axis = input_shape_0.size() - 1;
                    NNFUSION_CHECK(axis == input_shape_0.size() - 1);

                    auto code = nnfusion::op::create_code_from_template(
                        R"(
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= @groups@)
        return;
    for (int i = 0; i < @depth@; ++i)
        output0[idx * @depth@ + i] = @off_value@;
    output0[idx * @depth@ + (int)input0[idx]] = @on_value@;
)",
                        {
                            {"groups", groups},
                            {"depth", cfg["depth"]},
                            {"off_value", cfg["off_value"]},
                            {"on_value", cfg["on_value"]},
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
                    GENERIC_OP_LOGGING();

                    LanguageUnit_p _lu(new LanguageUnit(get_function_name() + "_dep"));
                    _lu->require(header::cuda);
                    return _lu;
                }

                void set_launch_config() override
                {
                    GENERIC_OP_LOGGING();

                    m_gridDim = dim3((groups + 63) / 64, 1, 1);
                    m_blockDim = dim3(64, 1, 1);
                }
            };
        } // namespace cuda
    }     // namespace kernels
} // namespace nnfusion

using namespace nnfusion;
using namespace nnfusion::kernels;

REGISTER_KERNEL_EMITTER(
    "OneHot",                                                                     // op_name
    Device(CUDA_GPU).TypeConstraint(element::f32).Tag("cuda_kernel").Priority(2), // attrs
    cuda::OneHot)                                                                 // constructor

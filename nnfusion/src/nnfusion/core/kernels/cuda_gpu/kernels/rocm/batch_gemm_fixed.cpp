// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "../../cuda_emitter.hpp"
#include "../../cuda_langunit.hpp"
#include "nnfusion/core/operators/generic_op/generic_op.hpp"

DEFINE_bool(frocm_fixed_kernels, true, "Enable Fixed kernel in ROCm codegen.");

namespace nnfusion
{
    namespace kernels
    {
        namespace cuda
        {
            class BatchGemmFixed : public CudaEmitter
            {
                shared_ptr<nnfusion::op::GenericOp> generic_op;

            public:
                BatchGemmFixed(shared_ptr<KernelContext> ctx)
                    : CudaEmitter(ctx)
                    , generic_op(
                          static_pointer_cast<nnfusion::op::GenericOp>(ctx->gnode->get_op_ptr()))
                {
                    GENERIC_OP_LOGGING();
                }

                LanguageUnit_p emit_function_body() override
                {
                    bool using_fixed = FLAGS_frocm_fixed_kernels;
                    if (!using_fixed)
                        return nullptr;

                    GENERIC_OP_LOGGING();
                    auto& ctx = m_context;

                    bool transA = generic_op->localOpConfig.getRoot()["adj_x"]["b"];
                    bool transB = generic_op->localOpConfig.getRoot()["adj_y"]["b"];

                    if (transA || transB)
                        return nullptr;

                    if (ctx->outputs[0]->get_element_type().c_type_string() != "float")
                        return nullptr;

                    nnfusion::Shape input_shape_0 = ctx->inputs[0]->get_shape();
                    nnfusion::Shape input_shape_1 = ctx->inputs[1]->get_shape();
                    if (input_shape_0.size() != input_shape_1.size())
                        return nullptr;

                    std::reverse(input_shape_0.begin(), input_shape_0.end());
                    std::reverse(input_shape_1.begin(), input_shape_1.end());
                    size_t batch_0 = 1, batch_1 = 1;
                    for (int i = 2; i < input_shape_0.size(); ++i)
                        batch_0 *= input_shape_0[i], batch_1 *= input_shape_1[i];

                    NNFUSION_CHECK(input_shape_0.size() >= 2 && input_shape_1.size() >= 2);
                    NNFUSION_CHECK(batch_0 == batch_1);
                    input_shape_0.resize(2), input_shape_1.resize(2);
                    input_shape_0.push_back(batch_0), input_shape_1.push_back(batch_1);
                    std::reverse(input_shape_0.begin(), input_shape_0.end());
                    std::reverse(input_shape_1.begin(), input_shape_1.end());

                    NNFUSION_CHECK(input_shape_0[2] == input_shape_1[1]);

                    std::string templ;
                    if (input_shape_0 == nnfusion::Shape({16, 512, 512}) &&
                        input_shape_1 == nnfusion::Shape({16, 512, 64}))
                    {
                        templ =
                            "rocm_adapter/fixed_kernels/batch_gemm/"
                            "batch_matmul_autotvm_NN_16x512x512x64.h.in";
                        m_gridDim = dim3(1, 16, 16);
                        m_blockDim = dim3(32, 8, 1);
                    }
                    else if (input_shape_0 == nnfusion::Shape({16, 512, 64}) &&
                             input_shape_1 == nnfusion::Shape({16, 64, 512}))
                    {
                        templ =
                            "rocm_adapter/fixed_kernels/batch_gemm/"
                            "batch_matmul_autotvm_NN_16x512x64x512.h.in";
                        m_gridDim = dim3(16, 8, 16);
                        m_blockDim = dim3(8, 32, 1);
                    }
                    else
                        return nullptr;

                    LanguageUnit_p _lu(new LanguageUnit(get_function_name()));
                    auto& lu = *_lu;
                    lu.block_begin();
                    lu << nnfusion::codegen::get_content_from_templates(templ) << "\n";
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

                void set_launch_config() override { GENERIC_OP_LOGGING(); }
            };
        } // namespace cuda
    }     // namespace kernels
} // namespace nnfusion

using namespace nnfusion;
using namespace nnfusion::kernels;

REGISTER_KERNEL_EMITTER("BatchMatMul",                                             // op_name
                        Device(ROCM_GPU).TypeConstraint(element::f32).Priority(4), // attrs
                        cuda::BatchGemmFixed)                                      // constructor

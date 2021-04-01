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
            class ApplyGradientDescent : public BlockCudaEmitter
            {
                shared_ptr<nnfusion::op::GenericOp> generic_op;
                size_t threads;

            public:
                ApplyGradientDescent(shared_ptr<KernelContext> ctx)
                    : BlockCudaEmitter(ctx)
                    , generic_op(
                          static_pointer_cast<nnfusion::op::GenericOp>(ctx->gnode->get_op_ptr()))
                {
                    threads = ctx->inputs[0]->size(false);
                    if (!ctx->annotations)
                        ctx->annotations = std::make_shared<Annotations>();
                    ctx->annotations->add_in_place_oi_pair({0, 0, true, true});
                }

                LanguageUnit_p emit_function_body() override
                {
                    auto& cfg = generic_op->localOpConfig.getRoot();
                    float lr = cfg["learning_rate"].is_null() ? 0.001 : (float)cfg["learning_rate"];

                    LanguageUnit_p _lu(new LanguageUnit(get_function_name()));
                    auto& lu = *_lu;
                    lu << "const uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;\n";
                    lu << "if (tid < " << threads << ")\n";
                    lu.block_begin();
                    {
                        lu << "output0[tid] -= " << lr << " * input1[tid];\n";
                    }
                    lu.block_end();
                    return _lu;
                }

                LanguageUnit_p emit_dependency() override
                {
                    LanguageUnit_p _lu(new LanguageUnit(get_function_name() + "_dep"));
                    _lu->require(header::cuda);
                    return _lu;
                }

                void set_launch_config() override
                {
                    uint32_t block_size_x = 512;
                    size_t block_cnt = align_to_block_size(threads, block_size_x);

                    m_gridDim = dim3(block_cnt, 1, 1);
                    m_blockDim = dim3(block_size_x, 1, 1);
                }
            };
        }
    }
}

using namespace nnfusion;
using namespace nnfusion::kernels;

REGISTER_KERNEL_EMITTER(
    "ApplyGradient",
    Device(CUDA_GPU).TypeConstraint(element::f32).Tag("cuda_kernel").Priority(2),
    cuda::ApplyGradientDescent)
REGISTER_KERNEL_EMITTER(
    "ApplyGradientDescent",
    Device(CUDA_GPU).TypeConstraint(element::f32).Tag("cuda_kernel").Priority(2),
    cuda::ApplyGradientDescent)

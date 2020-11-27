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
            class ApplyMomentum : public BlockCudaEmitter
            {
                shared_ptr<nnfusion::op::GenericOp> generic_op;
                size_t threads;

            public:
                ApplyMomentum(shared_ptr<KernelContext> ctx)
                    : BlockCudaEmitter(ctx)
                    , generic_op(
                          static_pointer_cast<nnfusion::op::GenericOp>(ctx->gnode->get_op_ptr()))
                {
                    threads = ctx->inputs[0]->size(false);
                    if (!ctx->annotations)
                        ctx->annotations = std::make_shared<Annotations>();
                    // force the output address == input address
                    ctx->annotations->add_in_place_oi_pair({0, 0, false});
                }

                LanguageUnit_p emit_function_body() override
                {
                    auto& cfg = generic_op->localOpConfig.getRoot();
                    bool use_nesterov = (bool)cfg["use_nesterov"];
                    float lr = cfg["lr"].is_null() ? 0.001 : (float)cfg["lr"];
                    float momentum = cfg["momentum"].is_null() ? 0.001 : (float)cfg["momentum"];
                    LanguageUnit_p _lu(new LanguageUnit(get_function_name()));
                    auto& lu = *_lu;
                    lu << "uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;\n";
                    lu << "if(tid < " << threads << ")\n";
                    lu.block_begin();
                    {
                        lu << "input1[tid] = input1[tid] * " << momentum << " + input2[tid];\n";
                        if (use_nesterov)
                            lu << "input0[tid] -= input2[tid] * " << lr << " + input1[tid] * "
                               << momentum << " * " << lr << ";\n";
                        else
                            lu << "input0[tid] -= input1[tid] * " << lr << ";\n";
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
    "ApplyMomentum",
    Device(CUDA_GPU).TypeConstraint(element::f32).Tag("cuda_kernel").Priority(2),
    cuda::ApplyMomentum)

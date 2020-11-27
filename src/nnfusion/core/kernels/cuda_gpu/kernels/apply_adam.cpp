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
            class ApplyAdam : public BlockCudaEmitter
            {
                shared_ptr<nnfusion::op::GenericOp> generic_op;
                size_t threads;
                float beta1, beta2, epsilon, beta1_pow, beta2_pow;

            public:
                ApplyAdam(shared_ptr<KernelContext> ctx)
                    : BlockCudaEmitter(ctx)
                    , generic_op(
                          static_pointer_cast<nnfusion::op::GenericOp>(ctx->gnode->get_op_ptr()))
                {
                    threads = ctx->inputs[0]->size(false);

                    auto& cfg = generic_op->localOpConfig.getRoot();
                    epsilon = cfg["epsilon"];
                    beta1 = cfg["beta1"];
                    beta1_pow = cfg["beta1_pow"];
                    beta2 = cfg["beta2"];
                    beta2_pow = cfg["beta2_pow"];

                    if (!ctx->annotations)
                        ctx->annotations = std::make_shared<Annotations>();
                    ctx->annotations->add_in_place_oi_pair({0, 0, false}); // inplace step update
                }

                LanguageUnit_p emit_function_body() override
                {
                    GENERIC_OP_LOGGING();
                    LanguageUnit_p _lu(new LanguageUnit(get_function_name()));
                    auto& lu = *_lu;

                    lu << "auto lr = input0[0];\n";
                    lu << "uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;\n";
                    lu << "if (i < " << threads << ")\n";
                    lu.block_begin();
                    {
                        auto code = op::create_code_from_template(
                            R"(
auto m = input1[i] * @beta1@ + (1 - @beta1@) * input4[i];
auto v = input2[i] * @beta2@ + (1 - @beta2@) * input4[i] * input4[i];
float denom = sqrt(v) + @epsilon@;
float update = (m/ denom) + input2[i];
float delta = -lr * update;

input0[i] = input0[i] + delta;
input1[i] = m;
input2[i] = v;

if(i == 0)
    input3[0] = input3[0] - @beta2@/(1-@beta1@);
)",
                            {{"beta1", beta1}, {"beta2", beta2}, {"epsilon", epsilon}});

                        lu << code;
                    }
                    lu.block_end();
                    lu << "if (i == 0)\n";
                    lu.block_begin();
                    {
                        // update step
                        lu << "output0[0] = input1[0] + 1;\n";
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
    "ApplyAdam",
    Device(CUDA_GPU).TypeConstraint(element::f32).Tag("cuda_kernel").Priority(2),
    cuda::ApplyAdam)
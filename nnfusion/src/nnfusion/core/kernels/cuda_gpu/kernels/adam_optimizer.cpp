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
            class AdamOptimizer : public BlockCudaEmitter
            {
                shared_ptr<nnfusion::op::GenericOp> generic_op;
                size_t threads;
                float lambda, epsilon, alpha, beta;

            public:
                AdamOptimizer(shared_ptr<KernelContext> ctx)
                    : BlockCudaEmitter(ctx)
                    , generic_op(
                          static_pointer_cast<nnfusion::op::GenericOp>(ctx->gnode->get_op_ptr()))
                {
                    threads = ctx->inputs[2]->size(false);

                    auto& cfg = generic_op->localOpConfig.getRoot();
                    lambda = cfg["lambda"];
                    epsilon = cfg["epsilon"];
                    alpha = cfg["alpha"];
                    beta = cfg["beta"];
                    if (!ctx->annotations)
                        ctx->annotations = std::make_shared<Annotations>();
                    // TODO: we use inplace_annotation to implement the reference_tensor, i.e., the
                    // output 0 shares the same address with input 0
                    // need to add a new annotation type or ref_tensor mechanism in the future
                    ctx->annotations->add_in_place_oi_pair({0, 1, false}); // inplace step update
                    ctx->annotations->add_in_place_oi_pair({1, 4, false}); // inplace m1
                    ctx->annotations->add_in_place_oi_pair({2, 5, false}); // inplace m2
                    ctx->annotations->add_in_place_oi_pair({3, 2, false}); // inplace weight
                    ctx->annotations->add_in_place_oi_pair({4, 3, false}); // inplace grad
                }

                LanguageUnit_p emit_function_body() override
                {
                    GENERIC_OP_LOGGING();
                    LanguageUnit_p _lu(new LanguageUnit(get_function_name()));
                    auto& lu = *_lu;

                    lu << m_context->dtypes[0] << " lr = input0[0];\n";
                    lu << "uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;\n";
                    lu << "if (i < " << threads << ")\n";
                    lu.block_begin();
                    {
                        auto code = op::create_code_from_template(
                            R"(
float m1_new = @alpha@ * input4[i] + ((1 - @alpha@) * input3[i]);
float m2_new = @beta@ * input5[i] + ((1 - @beta@) * input3[i] * input3[i]);

float denom = sqrt(m2_new) + @epsilon@;
float update = (m1_new / denom) + (@lambda@ * input2[i]);
float delta = -lr * update;

output1[i] = m1_new;
output2[i] = m2_new;
output3[i] = input2[i] + delta;
//output4[i] = delta;
)",
                            {{"alpha", alpha},
                             {"beta", beta},
                             {"epsilon", epsilon},
                             {"lambda", lambda}});

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

REGISTER_KERNEL_EMITTER("AdamOptimizer",
                        Device(CUDA_GPU).TypeConstraint(element::f32).Tag("cuda_kernel"),
                        cuda::AdamOptimizer)
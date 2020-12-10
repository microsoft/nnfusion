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
            class CrossEntropyFwdBwdWithSoftmaxBwdLarge : public BlockCudaEmitter
            {
                shared_ptr<nnfusion::op::GenericOp> generic_op;
                size_t threads;
                size_t class_num;

            public:
                CrossEntropyFwdBwdWithSoftmaxBwdLarge(shared_ptr<KernelContext> ctx)
                    : BlockCudaEmitter(ctx)
                    , generic_op(
                          static_pointer_cast<nnfusion::op::GenericOp>(ctx->gnode->get_op_ptr()))
                {
                    const nnfusion::Shape& input_shape = m_context->inputs[0]->get_shape();
                    auto shape_size = nnfusion::shape_size(input_shape);
                    threads = shape_size;
                    class_num = input_shape.back();
                }

                LanguageUnit_p emit_function_body() override
                {
                    GENERIC_OP_LOGGING();
                    LanguageUnit_p _lu(new LanguageUnit(get_function_name()));
                    auto& lu = *_lu;

                    lu << "uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;\n";
                    lu << "if (i < " << threads << ")\n";
                    lu.block_begin();
                    {
                        auto code = op::create_code_from_template(R"(
output0[i] = input0[i] + ((int)input1[i / @class_num@] == i % @class_num@ ? -1 : 0);
                        )",
                                                                  {{"class_num", class_num}});

                        lu << code;
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

REGISTER_KERNEL_EMITTER("CrossEntropyFwdBwdWithSoftmaxBwdLarge",
                        Device(CUDA_GPU).TypeConstraint(element::f32).Tag("cuda_kernel"),
                        cuda::CrossEntropyFwdBwdWithSoftmaxBwdLarge)
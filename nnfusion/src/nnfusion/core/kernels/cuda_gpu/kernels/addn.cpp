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
            class AddN : public BlockCudaEmitter
            {
                shared_ptr<nnfusion::op::GenericOp> generic_op;
                size_t threads;
                size_t input_count;
                nnfusion::element::Type dtype;

            public:
                AddN(shared_ptr<KernelContext> ctx)
                    : BlockCudaEmitter(ctx)
                    , generic_op(
                          static_pointer_cast<nnfusion::op::GenericOp>(ctx->gnode->get_op_ptr()))
                {
                    threads = ctx->outputs.front()->size(false);
                    input_count = ctx->inputs.size();
                    dtype = nnfusion::element::Type(ctx->outputs[0]->get_element_type());
                }

                LanguageUnit_p emit_function_body() override
                {
                    LanguageUnit_p _lu(new LanguageUnit(get_function_name()));
                    auto& lu = *_lu;

                    // Emit a function body will emmit the code inside the function:
                    //  global void function(type* input0, type* input1, ..., type* output0, type* output1)
                    //  {
                    //      <emit_function_body() will generate code inside here>
                    //  }

                    // Since add n is a more complex elementwise operator, so we
                    // issue *this->threads* threads to calculate by each element.

                    // Here are two ways to generate your code
                    // First: using string stream
                    lu << "uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;\n";
                    lu << "if(tid < " << threads << ")\n";
                    lu.block_begin();
                    {
                        lu << dtype.c_type_string() << " accum = 0;\n";
                        for (size_t i = 0; i < input_count; i++)
                            lu << "accum += input" << i << "[tid];\n";
                        lu << "output0[tid] = accum;\n";
                    }
                    lu.block_end();

                    // Second: using template. You can read one_hot.cpp for detail, for this case
                    // you cannot use template since we have uncertain amount of "accum += input$i[$threadid]".

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
    "AddN",
    Device(CUDA_GPU).TypeConstraint(element::f32).Tag("cuda_kernel").Priority(2),
    cuda::AddN)

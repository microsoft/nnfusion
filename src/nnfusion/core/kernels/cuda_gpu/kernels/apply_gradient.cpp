// Microsoft (c) 2019, NNFusion Team

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
            class ApplyGradient : public BlockCudaEmitter
            {
                shared_ptr<nnfusion::op::GenericOp> generic_op;
                size_t threads;

            public:
                ApplyGradient(shared_ptr<KernelContext> ctx)
                    : BlockCudaEmitter(ctx)
                    , generic_op(
                          static_pointer_cast<nnfusion::op::GenericOp>(ctx->gnode->get_op_ptr()))
                {
                    threads = ctx->inputs[0]->size(false);
                }

                LanguageUnit_p emit_function_body() override
                {
                    auto& cfg = generic_op->localOpConfig.getRoot();
                    float lr = cfg["learning_rate"].is_null() ? 0.001 : (float)cfg["learning_rate"];

                    LanguageUnit_p _lu(new LanguageUnit(get_function_name()));
                    auto& lu = *_lu;
                    lu << "// float lr = " << lr << ";\n";
                    lu << "// uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;\n";
                    lu << "// if(tid < " << threads << ")\n";
                    lu.block_begin();
                    {
                        // output the gradient for debugging
                        lu << "// output0[tid] = input1[tid];\n";
                        lu << "// input0[tid] = input0[tid] - lr * input1[tid];\n";
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

REGISTER_KERNEL_EMITTER("ApplyGradient",
                        Device(CUDA_GPU).TypeConstraint(DT_FLOAT).Tag("cuda_kernel"),
                        cuda::ApplyGradient)

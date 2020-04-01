// Microsoft (c) 2019, NNFusion Team

// This is the 2rd-generation of kernel definition, recommend to extend new ops with this style
// Changes needed for creating an new kernel with 2rd generation style:
//
// files to change:
//   [a] ./new_kernel_0.cpp
//   [b] ../../../ops/op_define/new_op_0.cpp

#include "../cuda_emitter.hpp"
#include "../cuda_langunit.hpp"
#include "nnfusion/core/operators/generic_op/generic_op.hpp"

namespace nnfusion
{
    namespace kernels
    {
        namespace cuda
        {
            class StopGradient : public CudaLibEmitter
            {
                shared_ptr<nnfusion::op::GenericOp> generic_op;

            public:
                StopGradient(shared_ptr<KernelContext> ctx)
                    : CudaLibEmitter(ctx)
                    , generic_op(
                          static_pointer_cast<nnfusion::op::GenericOp>(ctx->gnode->get_op_ptr()))
                {
                    GENERIC_OP_LOGGING();
                }

                LanguageUnit_p emit_function_body() override
                {
                    GENERIC_OP_LOGGING();

                    const nnfusion::Shape& input_shape_0 = m_context->inputs[0]->get_shape();
                    size_t mul_cnt = 1;
                    for (auto& it : input_shape_0)
                        mul_cnt *= it;

                    auto code = nnfusion::op::create_code_from_template(
                        R"(
                        CUDA_SAFE_CALL(cudaMemcpyAsync(output0, input0, @size@, cudaMemcpyDeviceToDevice, stream));
                    )",
                        {
                            {"size",
                             std::to_string(mul_cnt) + "LU * sizeof(" + m_context->dtypes[0] + ")"},
                        });

                    LanguageUnit_p _lu(new LanguageUnit(get_function_name()));
                    auto& lu = *_lu;
                    // function signature:
                    // extern "C" __global__ void kernel(m_context->dtypes[0]* input0, m_context->dtypes[0]* input1, m_context->dtypes[2]* output0)
                    lu.block_begin();
                    lu << code << "\n";
                    lu.block_end();
                    return _lu;
                }

                LanguageUnit_p emit_dependency() override
                {
                    GENERIC_OP_LOGGING();

                    LanguageUnit_p _lu_header(new LanguageUnit(get_function_name() + "_dep"));
                    _lu_header->require(header::cuda);
                    _lu_header->require(macro::CUDA_SAFE_CALL);
                    return _lu_header;
                }
            };
        } // namespace cuda
    }     // namespace kernels
} // namespace nnfusion

// Register Pad kernel emitter

using namespace nnfusion;
using namespace nnfusion::kernels;

REGISTER_KERNEL_EMITTER("StopGradient",                                               // op_name
                        Device(CUDA_GPU).TypeConstraint(DT_FLOAT).Tag("cuda_kernel"), // attrs
                        cuda::StopGradient)                                           // constructor

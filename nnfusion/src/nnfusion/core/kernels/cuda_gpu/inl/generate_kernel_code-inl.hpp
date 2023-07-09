// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

namespace nnfusion
{
    namespace kernels
    {
        namespace cuda
        {
            class __KernelUniqueClassName__ : public BlockCudaEmitter
            {
                shared_ptr<nnfusion::op::GenericOp> generic_op;

            public:
                __KernelUniqueClassName__(shared_ptr<KernelContext> ctx)
                    : BlockCudaEmitter(ctx)
                    , generic_op(
                          static_pointer_cast<nnfusion::op::GenericOp>(ctx->gnode->get_op_ptr()))
                {
                }

                LanguageUnit_p emit_function_body() override
                {
                    GENERIC_OP_LOGGING();

                    size_t num_in = m_context->inputs.size(), num_out = m_context->outputs.size();
                    std::vector<nnfusion::Shape> input_shapes, output_shapes;
                    for (int i = 0; i < num_in; ++i)
                        input_shapes.push_back(m_context->inputs[i]->get_shape());
                    for (int i = 0; i < num_out; ++i)
                        output_shapes.push_back(m_context->outputs[i]->get_shape());

                    auto res = generate_kernel_code(
                        input_shapes, output_shapes, generic_op->localOpConfig.getRoot());
                    if (res.is_null())
                        return nullptr;

                    m_blockDim =
                        dim3(res["block_dim"][0], res["block_dim"][1], res["block_dim"][2]);
                    m_gridDim = dim3(res["grid_dim"][0], res["grid_dim"][1], res["grid_dim"][2]);

                    LanguageUnit_p _lu(new LanguageUnit(get_function_name()));
                    auto& lu = *_lu;

                    lu.block_begin();
                    lu << (std::string)res["source_code"] << "\n";
                    lu.block_end();
                    return _lu;
                }

                LanguageUnit_p emit_dependency() override
                {
                    LanguageUnit_p _lu(new LanguageUnit(get_function_name() + "_dep"));
                    _lu->require(header::cuda);
                    return _lu;
                }

                void set_launch_config() override {}
            };
        } // namespace cuda
    }     // namespace kernels
} // namespace nnfusion

using namespace nnfusion;
using namespace nnfusion::kernels;

REGISTER_KERNEL_EMITTER(__KernelOpType__,                                                 // op_name
                        Device(CUDA_GPU).TypeConstraint(element::f32).Tag("cuda_kernel"), // attrs
                        cuda::__KernelUniqueClassName__) // constructor

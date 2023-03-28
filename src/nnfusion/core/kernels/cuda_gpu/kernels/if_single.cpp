#include "../cuda_emitter.hpp"
#include "../cuda_langunit.hpp"
#include "nnfusion/core/operators/op_define/if_single.hpp"

namespace nnfusion
{
    namespace kernels
    {
        namespace cuda
        {

            class IfSingleBlockCuda : public BlockCudaEmitter
            {
            public:
                IfSingleBlockCuda(shared_ptr<KernelContext> ctx): BlockCudaEmitter(ctx), m_kernel_ctx(ctx) {
                    m_op = static_pointer_cast<nnfusion::op::IfSingle>(m_kernel_ctx->gnode->get_op_ptr());
                    NNFUSION_CHECK_NOT_NULLPTR(m_op);
                    m_inner_node = m_op->get_inner_fake_node(m_kernel_ctx->gnode);
                    NNFUSION_CHECK((*m_inner_node)["Kernel_Selection_Result"].is_valid());
                    auto inner_emitted_kernel =
                        (*m_inner_node)["Kernel_Selection_Result"].as<pair<NNFusion_DeviceType, KernelEmitter::Pointer>>();
                    m_inner_kernel_emitter = std::dynamic_pointer_cast<BlockCudaEmitter>(inner_emitted_kernel.second);
                    set_num_local_thread_sync(m_inner_kernel_emitter->get_num_local_thread_sync());
                    set_shared_memory_size(m_inner_kernel_emitter->get_shared_memory_size());
                }

                LanguageUnit_p emit_function_body() {
                    if (m_inner_kernel_emitter == nullptr) {
                        return nullptr;
                    }
                    LanguageUnit_p _lu(new LanguageUnit(get_function_name()));
                    auto& lu = *_lu;
                    std::vector<std::string> input_names;
                    std::vector<std::string> output_names;
                    for (size_t i = 1; i < m_kernel_ctx->inputs.size(); i++) {
                        input_names.push_back("input" + to_string(i));
                    }
                    for (size_t i = 0; i < m_kernel_ctx->outputs.size(); i++) {
                        output_names.push_back("output" + to_string(i));
                    }
                    lu << "if ((*input0) == " << m_op->get_is_then_branch()  << ")\n";
                    lu.block_begin();
                    lu << m_inner_kernel_emitter->get_function_name() << "_block_kernel(" << join(input_names, ", ") << ", " << join(output_names, ", ") << ", thread_id, block_id, shared_buffer);\n";
                    lu.block_end();
                    return _lu;
                }

                LanguageUnit_p emit_dependency() override {
                    NNFUSION_CHECK((*m_inner_node)["Kernel_Selection_Result"].is_valid());
                    LanguageUnit_p _lu(new LanguageUnit(get_function_name() + "_dep"));
                    _lu->require(m_inner_kernel_emitter->emit_block_kernel());
                    _lu->require(m_inner_kernel_emitter->emit_dependency());
                    return _lu;
                }

                void set_launch_config() override {
                    if (m_inner_kernel_emitter == nullptr) {
                        return;
                    }
                    m_gridDim = m_inner_kernel_emitter->get_grid_dim();
                    m_blockDim = m_inner_kernel_emitter->get_block_dim();
                }
            private:
                shared_ptr<KernelContext> m_kernel_ctx;
                shared_ptr<graph::GNode> m_inner_node;
                shared_ptr<BlockCudaEmitter> m_inner_kernel_emitter;
                shared_ptr<nnfusion::op::IfSingle> m_op;
            };
        } // namespace cuda
    }     // namespace kernels
} // namespace nnfusion

using namespace nnfusion;
using namespace nnfusion::kernels;
REGISTER_KERNEL_EMITTER(
    "IfSingle",                                                                    // op_name
    Device(CUDA_GPU).TypeConstraint(element::f32).Tag("cuda_kernel").Priority(2), // attrs
    cuda::IfSingleBlockCuda)                                                              // constructor

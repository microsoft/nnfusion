#include "../cuda_emitter.hpp"
#include "../cuda_langunit.hpp"

namespace nnfusion
{
    namespace kernels
    {
        namespace cuda
        {
            class D2H : public CudaLibEmitter
            {
            public:
                D2H(shared_ptr<KernelContext> ctx) : CudaLibEmitter(ctx) {
                    m_shape = ctx->gnode->get_input_shape(0);
                    m_type = ctx->gnode->get_input_element_type(0);
                }

                LanguageUnit_p emit_function_body() override {
                    LanguageUnit_p _lu(new LanguageUnit(get_function_name()));
                    auto& lu = *_lu;
                    lu << "CUDA_SAFE_CALL(cudaMemcpyAsync(output0, input0, " << m_type.size() * shape_size(m_shape) << ", cudaMemcpyDeviceToHost));\n";
                    lu << "CUDA_SAFE_CALL(cudaDeviceSynchronize());\n";
                    return _lu;
                }

                LanguageUnit_p emit_function_call() override {
                    auto gnode = m_context->gnode;
                    LanguageUnit_p _lu(new LanguageUnit(this->m_kernel_name + "_call"));
                    auto& lu = *_lu;
                    vector<string> names;
                    NNFUSION_CHECK(m_context->input_names.size() == 1);
                    NNFUSION_CHECK(m_context->output_names.size() == 1);
                    NNFUSION_CHECK(m_context->tensor_names.size() == 0);
                    for (auto name: m_context->input_names) names.push_back(name);
                    for (auto name: m_context->output_names) names.push_back(name + "_cpu");
                    lu << "(" << join(names, ", ") << ");\n";
                    return _lu;
                }

                LanguageUnit_p emit_dependency() override {
                    LanguageUnit_p _lu(new LanguageUnit(get_function_name() + "_dep"));
                    return _lu;
                }

            private:
                nnfusion::Shape m_shape;
                nnfusion::element::Type m_type;
            };
        } // namespace cuda
    }     // namespace kernels
} // namespace nnfusion

using namespace nnfusion;
using namespace nnfusion::kernels;
REGISTER_KERNEL_EMITTER("d2h",                                                              //op_name
                        Device(CUDA_GPU).TypeConstraint(element::f32),                      //attrs
                        cuda::D2H);                                                         // constructor

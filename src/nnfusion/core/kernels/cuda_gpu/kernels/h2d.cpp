#include "../cuda_emitter.hpp"
#include "../cuda_langunit.hpp"

namespace nnfusion
{
    namespace kernels
    {
        namespace cuda
        {
            class H2D : public CudaLibEmitter
            {
            public:
                H2D(shared_ptr<KernelContext> ctx) : CudaLibEmitter(ctx) {
                    m_shape = ctx->gnode->get_input_shape(0);
                    m_type = ctx->gnode->get_input_element_type(0);
                }

                LanguageUnit_p emit_function_body() override {
                    LanguageUnit_p _lu(new LanguageUnit(get_function_name()));
                    auto& lu = *_lu;
                    lu << "CUDA_SAFE_CALL(cudaMemcpyAsync(output0, input0, " << m_type.size() * shape_size(m_shape) << ", cudaMemcpyDeviceToDevice)); // fake h2d \n";
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
REGISTER_KERNEL_EMITTER("h2d",                                                              //op_name
                        Device(CUDA_GPU).TypeConstraint(element::f32),                      //attrs
                        cuda::H2D);                                                         // constructor

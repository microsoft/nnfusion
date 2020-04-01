// Microsoft (c) 2019, NNFusion Team

#include "../../cuda_cudnn.hpp"
#include "../../cuda_emitter.hpp"
#include "../../cuda_langunit.hpp"
#include "nnfusion/core/operators/generic_op/generic_op.hpp"

namespace nnfusion
{
    namespace kernels
    {
        namespace cuda
        {
            class MemcpySlice : public CudaLibEmitter
            {
            public:
                MemcpySlice(shared_ptr<KernelContext> ctx)
                    : CudaLibEmitter(ctx)
                {
                    auto slice_op =
                        static_pointer_cast<nnfusion::op::Slice>(ctx->gnode->get_op_ptr());

                    input_shape = nnfusion::Shape(ctx->inputs[0]->get_shape());
                    output_shape = nnfusion::Shape(ctx->outputs[0]->get_shape());

                    input_type = ctx->inputs[0]->get_element_type().c_type_string();
                    output_type = ctx->outputs[0]->get_element_type().c_type_string();
                    lower_bounds = slice_op->get_lower_bounds();

                    input_strides = row_major_strides(input_shape);
                    output_strides = row_major_strides(output_shape);
                    slice_strides = slice_op->get_strides();
                }

                LanguageUnit_p emit_function_body() override
                {
                    auto& ctx = m_context;
                    auto input_shape = nnfusion::Shape(ctx->inputs[0]->get_shape());
                    auto output_shape = nnfusion::Shape(ctx->outputs[0]->get_shape());

                    auto op_config =
                        static_pointer_cast<nnfusion::op::Slice>(ctx->gnode->get_op_ptr());
                    auto lower_bounds = op_config->get_lower_bounds();
                    auto slice_strides = op_config->get_strides();
#if 0 // for Debug
                    for (int i = 0; i < input_shape.size(); ++i) printf("%d ", (int)input_shape[i]); puts("");
                    for (int i = 0; i < output_shape.size(); ++i) printf("%d ", (int)output_shape[i]); puts("");
                    for (int i = 0; i < lower_bounds.size(); ++i) printf("%d ", (int)lower_bounds[i]); puts("");
                    for (int i = 0; i < slice_strides.size(); ++i) printf("%d ", (int)slice_strides[i]); puts("");
                    for (int i = 0; i < input_strides.size(); ++i) printf("%d ", (int)input_strides[i]); puts("");
                    for (int i = 0; i < output_strides.size(); ++i) printf("%d ", (int)output_strides[i]); puts("");
                    puts("======");
#endif

                    for (int i = 0; i < slice_strides.size(); ++i)
                        if (slice_strides[i] != 1)
                            return nullptr;
                    for (int i = 1; i < lower_bounds.size(); ++i)
                        if (lower_bounds[i] != 0)
                            return nullptr;
                    size_t num_ele = 1, num_out = 1;
                    for (int i = 0; i < input_shape.size(); ++i)
                        num_ele *= input_shape[i];
                    for (int i = 0; i < output_shape.size(); ++i)
                        num_out *= output_shape[i];
                    if (num_ele != num_out * input_shape[0])
                        return nullptr;
                    if (ctx->outputs[0]->get_element_type().c_type_string() != "float")
                        return nullptr;

                    size_t offset = num_out * lower_bounds[0];

                    std::string code = nnfusion::op::create_code_from_template(
                        R"(
	CUDA_SAFE_CALL(hipMemcpyAsync(output0, input0 + @offset@LU, @size@, hipMemcpyDeviceToDevice, stream));
)",
                        {{"offset", offset}, {"size", num_out * sizeof(float)}});

                    LanguageUnit_p _lu(new LanguageUnit(get_function_name()));
                    auto& lu = *_lu;
                    lu << code << "\n";
                    return _lu;
                }
                LanguageUnit_p emit_dependency() override
                {
                    LanguageUnit_p _lu(new LanguageUnit(get_function_name() + "_dep"));

                    _lu->require(header::cuda);

                    return _lu;
                }

            private:
                shared_ptr<KernelContext> kernel_ctx;

                nnfusion::Shape input_shape, output_shape, lower_bounds;
                nnfusion::Shape input_strides, output_strides, slice_strides;
                string input_type, output_type;
            };
        } // namespace cuda
    }     // namespace kernels
} // namespace nnfusion

using namespace nnfusion;
using namespace nnfusion::kernels;

REGISTER_KERNEL_EMITTER("Slice",                                                     // op_name
                        Device(ROCM_GPU).TypeConstraint(DT_FLOAT).Tag("PRIORITY_1"), // attrs
                        cuda::MemcpySlice)                                           // constructor

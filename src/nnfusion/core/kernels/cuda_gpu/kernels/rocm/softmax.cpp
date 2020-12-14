// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "../../cuda_emitter.hpp"
#include "../../cuda_langunit.hpp"
#include "nnfusion/core/operators/generic_op/generic_op.hpp"

namespace nnfusion
{
    namespace kernels
    {
        namespace cuda
        {
            class RocmSoftmax : public CudaLibEmitter
            {
            public:
                RocmSoftmax(shared_ptr<KernelContext> ctx)
                    : CudaLibEmitter(ctx)
                {
                }

                LanguageUnit_p emit_function_body() override
                {
                    nnfusion::Shape input_shape, output_shape;
                    nnfusion::AxisSet axes;
                    size_t height, width;
                    bool valid_inputs = true;

                    auto& ctx = m_context;
                    auto node =
                        static_pointer_cast<nnfusion::op::Softmax>(ctx->gnode->get_op_ptr());
                    input_shape = nnfusion::Shape(ctx->inputs[0]->get_shape());
                    output_shape = nnfusion::Shape(ctx->outputs[0]->get_shape());

                    // this kernel currently can only handle 2D matrix, thus we have to transfer a >2D tensor
                    // to 2D softmax
                    axes = node->get_axes();
                    std::vector<size_t> axes_flag(input_shape.size(), 0);
                    for (auto const& axis : axes)
                    {
                        axes_flag[axis] = 1;
                    }
                    height = 1;
                    width = 1;
                    int i = 0;
                    for (; i < axes_flag.size() && axes_flag[i] == 0; i++)
                    {
                        height *= input_shape[i];
                    }
                    for (; i < axes_flag.size(); i++)
                    {
                        if (axes_flag[i] == 0)
                        {
                            valid_inputs = false;
                            break;
                        }
                        width *= input_shape[i];
                    }

                    if (!valid_inputs)
                        return nullptr;

                    // function signature:
                    // extern "C" __global__ void kernel(m_context->dtypes[0]* input0, m_context->dtypes[1]* output0)

                    auto code = nnfusion::op::create_code_from_template(
                        R"(
    float alpha = 1.0f, beta = 0.0f;
    miopenTensorDescriptor_t desc;
    CUDNN_SAFE_CALL(miopenCreateTensorDescriptor(&desc));
    CUDNN_SAFE_CALL(miopenSet4dTensorDescriptor(desc, miopenFloat, @height@, @width@, 1, 1));
    CUDNN_SAFE_CALL(miopenSoftmaxForward(cudnn_handle, &alpha, desc, input0, &beta, desc, output0));
    CUDNN_SAFE_CALL(miopenDestroyTensorDescriptor(desc));
)",
                        {{"height", height}, {"width", width}});

                    LanguageUnit_p _lu(new LanguageUnit(get_function_name()));
                    auto& lu = *_lu;
                    lu << code << "\n";
                    return _lu;
                }

                LanguageUnit_p emit_dependency() override
                {
                    LanguageUnit_p _lu(new LanguageUnit(get_function_name() + "_dep"));
                    _lu->require(header::cuda);
                    //_lu->require(declaration::cudnn_handle);
                    _lu->require(macro::CUDNN_SAFE_CALL);
                    return _lu;
                }

                LanguageUnit_p emit_function_signature() override
                {
                    LanguageUnit_p _lu(new LanguageUnit(this->m_kernel_name + "_sig"));
                    auto& lu = *_lu;

                    vector<string> params;
                    for (size_t i = 0; i < m_context->inputs.size(); i++)
                    {
                        stringstream ss;
                        ss << m_context->inputs[i]->get_element_type().c_type_string() << "* ";
                        ss << "input" << i;
                        params.push_back(ss.str());
                    }

                    for (size_t i = 0; i < m_context->outputs.size(); i++)
                    {
                        stringstream ss;
                        ss << m_context->outputs[i]->get_element_type().c_type_string() << "* ";
                        ss << "output" << i;
                        params.push_back(ss.str());
                    }

                    for (size_t i = 0; i < m_context->tensors.size(); i++)
                    {
                        stringstream ss;
                        ss << m_context->tensors[i]->get_element_type().c_type_string() << "* ";
                        // defult name is: "persit0", "persist1" ...
                        ss << m_context->tensors[i]->get_name();
                        params.push_back(ss.str());
                    }

                    lu << "void "
                       << "(cudnnHandle_t cudnn_handle, " << join(params, ", ") << ")";
                    return _lu;
                }
                bool require_cudnn_handle() override { return true; }
            };
        } // namespace cuda
    }     // namespace kernels
} // namespace nnfusion

using namespace nnfusion;
using namespace nnfusion::kernels;

REGISTER_KERNEL_EMITTER(
    "Softmax",                                                                    // op_name
    Device(ROCM_GPU).TypeConstraint(element::f32).Tag("cuda_kernel").Priority(2), // attrs
    cuda::RocmSoftmax)                                                            // constructor

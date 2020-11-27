// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "../../cuda_emitter.hpp"
#include "../../cuda_langunit.hpp"
#include "nnfusion/core/operators/generic_op/generic_op.hpp"

DECLARE_bool(frocm_candidate_kernels);

namespace nnfusion
{
    namespace kernels
    {
        namespace cuda
        {
            class RocmBiasBroadcast : public CudaLibEmitter
            {
            public:
                RocmBiasBroadcast(shared_ptr<KernelContext> ctx)
                    : CudaLibEmitter(ctx)
                {
                }

                LanguageUnit_p emit_function_body() override
                {
                    bool using_fixed = FLAGS_frocm_candidate_kernels;
                    if (!using_fixed)
                        return nullptr;

                    auto& ctx = m_context;
                    auto input_shape = nnfusion::Shape(ctx->inputs[0]->get_shape());
                    auto output_shape = nnfusion::Shape(ctx->outputs[0]->get_shape());

                    auto node =
                        static_pointer_cast<nnfusion::op::Broadcast>(ctx->gnode->get_op_ptr());
                    auto axes = node->get_broadcast_axes();

                    size_t in_size = 1;
                    for (auto& it : input_shape)
                        in_size *= it;
                    if (in_size == 1)
                        return nullptr;
                    // only handle (1, C, 1, 1) to (N, C, H, W)
                    if (input_shape.size() != 1 || output_shape.size() != 4 || axes.size() != 3)
                        return nullptr;
                    if (!axes.count(0) || !axes.count(2) || !axes.count(3))
                        return nullptr;
                    if (input_shape[0] != output_shape[1])
                        return nullptr;

                    std::vector<size_t> input_format, output_format = output_shape;
                    NNFUSION_CHECK(output_format.size() <= 4);

                    int shape_iter = 0;
                    for (int i = 0; i < output_shape.size(); ++i)
                    {
                        if (axes.count(i))
                            input_format.push_back(1);
                        else
                        {
                            NNFUSION_CHECK(shape_iter < input_shape.size());
                            input_format.push_back(input_shape[shape_iter++]);
                            NNFUSION_CHECK(input_format.back() ==
                                           output_format[input_format.size() - 1]);
                        }
                    }
                    while (output_format.size() < 4)
                    {
                        input_format.push_back(1);
                        output_format.push_back(1);
                    }

                    NNFUSION_CHECK(m_context->dtypes[0] == "float");
                    NNFUSION_CHECK(m_context->dtypes[1] == "float");

                    std::string code = nnfusion::op::create_code_from_template(
                        R"(
    float alpha = 1.0f, beta = 0.0f;
    miopenTensorDescriptor_t in_desc, out_desc;
    CUDNN_SAFE_CALL(miopenCreateTensorDescriptor(&in_desc));
    CUDNN_SAFE_CALL(miopenCreateTensorDescriptor(&out_desc));
    CUDNN_SAFE_CALL(miopenSet4dTensorDescriptor(in_desc, miopenFloat, @in_0@, @in_1@, @in_2@, @in_3@));
    CUDNN_SAFE_CALL(miopenSet4dTensorDescriptor(out_desc, miopenFloat, @out_0@, @out_1@, @out_2@, @out_3@));
    CUDNN_SAFE_CALL(miopenOpTensor(cudnn_handle, miopenTensorOpAdd, &beta, out_desc, output0, &alpha, in_desc, input0, &beta, out_desc, output0));
    CUDNN_SAFE_CALL(miopenDestroyTensorDescriptor(in_desc));
    CUDNN_SAFE_CALL(miopenDestroyTensorDescriptor(out_desc));
)",
                        {
                            {"in_0", input_format[0]},
                            {"in_1", input_format[1]},
                            {"in_2", input_format[2]},
                            {"in_3", input_format[3]},
                            {"out_0", output_format[0]},
                            {"out_1", output_format[1]},
                            {"out_2", output_format[2]},
                            {"out_3", output_format[3]},
                        });

                    if (input_format == output_format)
                    {
                        size_t num_eles = 1;
                        for (auto& it : input_format)
                            num_eles *= it;
                        code = nnfusion::op::create_code_from_template(
                            R"(
	CUDA_SAFE_CALL(hipMemcpyDtoD(output0, input0, @num_eles@LU));
)",
                            {
                                {"num_eles", num_eles},
                            });
                    }

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

// Register Pad kernel emitter

using namespace nnfusion;
using namespace nnfusion::kernels;

REGISTER_KERNEL_EMITTER("Broadcast",                                               //op_name
                        Device(ROCM_GPU).TypeConstraint(element::f32).Priority(3), //attrs
                        cuda::RocmBiasBroadcast)                                   // constructor

// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "softmax.hpp"
#include "../cuda_cudnn.hpp"
#include "nnfusion/core/operators/generic_op/generic_op.hpp"

using namespace nnfusion;
using namespace nnfusion::kernels;

cuda::Softmax::Softmax(shared_ptr<KernelContext> ctx)
    : CudaLibEmitter(ctx)
{
    auto node = static_pointer_cast<nnfusion::op::Softmax>(ctx->gnode->get_op_ptr());
    input_shape = nnfusion::Shape(ctx->inputs[0]->get_shape());
    output_shape = nnfusion::Shape(ctx->outputs[0]->get_shape());
    dtype = m_context->inputs[0]->get_element_type();
    is_log_softmax = node->is_in_log_space();
    algorithm = node->is_in_log_space() ? "CUDNN_SOFTMAX_LOG" : "CUDNN_SOFTMAX_ACCURATE";
    auto axisset = node->get_axes();
    size_t axis = output_shape.size() - 1;
    if (axisset.size() == 1)
        axis = *std::begin(axisset);
    N = 1;
    D = 1;
    for (size_t i = 0; i < input_shape.size(); i++)
    {
        if (i < axis)
        {
            N *= input_shape[i];
        }
        else
        {
            D *= input_shape[i];
        }
    }
}

LanguageUnit_p
    cuda::Softmax::cudnn_tensor_descriptor_from_shape_for_softmax(const nnfusion::Shape& shape,
                                                                  string desc)
{
    LanguageUnit_p _lu(new LanguageUnit);
    auto& lu = *_lu;
    // element::Type type = m_context->inputs[0]->get_element_type();
    string data_type = cuda::get_cudnn_datatype(dtype);
    string tensor_format = "CUDNN_TENSOR_NCHW";
    lu << "cudnnTensorDescriptor_t " << desc << ";\n";
    lu << "CUDNN_SAFE_CALL(cudnnCreateTensorDescriptor(&" << desc << "));\n";

    size_t n = shape[0], reduce_axis;
    if (shape.size() > 1)
    {
        for (int i = 1; i < shape.size() - 1; ++i)
        {
            n *= shape[i];
        }
    }
    else
    {
        n = 1;
    }
    reduce_axis = shape.back();

    size_t pos = 0;
    std::array<int, 4> dimensions;
    dimensions[pos++] = static_cast<int>(n);
    dimensions[pos++] = 1;
    dimensions[pos++] = 1;
    dimensions[pos++] = static_cast<int>(reduce_axis);

    lu << "CUDNN_SAFE_CALL(cudnnSetTensor4dDescriptor(" << desc << ", " << tensor_format << ", "
       << data_type << ", " << dimensions[0] << ", " << dimensions[1] << ", " << dimensions[2]
       << ", " << dimensions[3] << "));\n";
    return _lu;
}

LanguageUnit_p cuda::Softmax::emit_function_body()
{
    LanguageUnit_p _lu(new LanguageUnit(get_function_name()));
    auto& lu = *_lu;

    if (D <= 1024 && D * dtype.size() <= 4096)
    {
        auto code = nnfusion::op::create_code_from_template(
            R"(
dispatch_softmax_forward<@dtype@, @dtype@, float, @is_log_softmax@>(stream, output0, input0, @D@, @D@, @N@);
    )",
            {{"dtype", (dtype == element::f16) ? "half" : "float"},
             {"is_log_softmax", is_log_softmax},
             {"D", D},
             {"N", N}});

        lu << code << "\n";
        return _lu;
    }

    auto input_tensor_desc =
        cudnn_tensor_descriptor_from_shape_for_softmax(input_shape, "input_tensor_desc");
    auto output_tensor_desc =
        cudnn_tensor_descriptor_from_shape_for_softmax(output_shape, "output_tensor_desc");
    if (input_tensor_desc == nullptr || output_tensor_desc == nullptr)
    {
        return nullptr;
    }
    lu << input_tensor_desc->get_code();
    lu << output_tensor_desc->get_code();
    lu << "const float alpha = 1.0;\n";
    lu << "const float beta = 0.0;\n";
    lu << "CUDNN_SAFE_CALL(cudnnSoftmaxForward(cudnn_handle, " << algorithm << ", "
       << "CUDNN_SOFTMAX_MODE_INSTANCE, "
       << "&alpha, "
       << "input_tensor_desc, "
       << "input0, "
       << "&beta, "
       << "output_tensor_desc, "
       << "output0));\n";
    lu << "CUDNN_SAFE_CALL(cudnnDestroyTensorDescriptor(input_tensor_desc));\n";
    lu << "CUDNN_SAFE_CALL(cudnnDestroyTensorDescriptor(output_tensor_desc));\n";

    return _lu;
}

LanguageUnit_p cuda::Softmax::emit_dependency()
{
    LanguageUnit_p _lu(new LanguageUnit(get_function_name() + "_dep"));
    _lu->require(header::cuda);
    _lu->require(header::cudnn);
    //_lu->require(declaration::cudnn_handle);
    _lu->require(macro::CUDNN_SAFE_CALL);
    if (D <= 1024 && D * dtype.size() <= 4096)
    {
        _lu->require(declaration::ort_softmax);
        declaration::ort_softmax->require(declaration::warp);
        _lu->require(header::limits);
    }
    return _lu;
}

LanguageUnit_p cuda::Softmax::emit_function_signature()
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
    if (D <= 1024 && D * dtype.size() <= 4096)
    {
        lu << "void "
           << "(cudaStream_t stream, " << join(params, ", ") << ")";
    }
    else
    {
        lu << "void "
           << "(cudnnHandle_t cudnn_handle, " << join(params, ", ") << ")";
    }
    return _lu;
}

REGISTER_KERNEL_EMITTER(
    "Softmax",                                                                     // op_name
    Device(CUDA_GPU).TypeConstraint(element::f32).Tag("cudnn_kernel").Priority(2), // attrs
    cuda::Softmax)                                                                 // constructor

cuda::SoftmaxGrad::SoftmaxGrad(shared_ptr<KernelContext> ctx)
    : CudaLibEmitter(ctx)
{
    auto node = static_pointer_cast<nnfusion::op::Softmax>(ctx->gnode->get_op_ptr());
    NNFUSION_CHECK(ctx->gnode->get_input_size() == 2);
    // TODO: we should get shape from gnode instead of op;
    NNFUSION_CHECK(ctx->inputs[0]->get_shape() == ctx->inputs[1]->get_shape());
    input_shape = nnfusion::Shape(ctx->inputs[0]->get_shape());
    output_shape = nnfusion::Shape(ctx->outputs[0]->get_shape());
    algorithm = node->is_in_log_space() ? "CUDNN_SOFTMAX_LOG" : "CUDNN_SOFTMAX_ACCURATE";
}

LanguageUnit_p
    cuda::SoftmaxGrad::cudnn_tensor_descriptor_from_shape_for_softmax(const nnfusion::Shape& shape,
                                                                      string desc)
{
    LanguageUnit_p _lu(new LanguageUnit);
    auto& lu = *_lu;
    string data_type = "CUDNN_DATA_FLOAT"; //cuda::get_cudnn_datatype(type);
    string tensor_format = "CUDNN_TENSOR_NCHW";
    lu << "cudnnTensorDescriptor_t " << desc << ";\n";
    lu << "CUDNN_SAFE_CALL(cudnnCreateTensorDescriptor(&" << desc << "));\n";

    size_t n = shape[0], reduce_axis;
    if (shape.size() > 1)
    {
        for (int i = 1; i < shape.size() - 1; ++i)
        {
            n *= shape[i];
        }
    }
    else
    {
        n = 1;
    }
    reduce_axis = shape.back();

    size_t pos = 0;
    std::array<int, 4> dimensions;
    dimensions[pos++] = static_cast<int>(n);
    dimensions[pos++] = 1;
    dimensions[pos++] = 1;
    dimensions[pos++] = static_cast<int>(reduce_axis);

    lu << "CUDNN_SAFE_CALL(cudnnSetTensor4dDescriptor(" << desc << ", " << tensor_format << ", "
       << data_type << ", " << dimensions[0] << ", " << dimensions[1] << ", " << dimensions[2]
       << ", " << dimensions[3] << "));\n";
    return _lu;
}

LanguageUnit_p cuda::SoftmaxGrad::emit_function_body()
{
    LanguageUnit_p _lu(new LanguageUnit(get_function_name()));
    auto& lu = *_lu;

    auto input_tensor_desc =
        cudnn_tensor_descriptor_from_shape_for_softmax(input_shape, "input_tensor_desc");
    auto output_tensor_desc =
        cudnn_tensor_descriptor_from_shape_for_softmax(output_shape, "output_tensor_desc");
    if (input_tensor_desc == nullptr || output_tensor_desc == nullptr)
    {
        return nullptr;
    }
    lu << input_tensor_desc->get_code();
    lu << output_tensor_desc->get_code();
    lu << "const float alpha = 1.0;\n";
    lu << "const float beta = 0.0;\n";
    lu << "CUDNN_SAFE_CALL(cudnnSoftmaxBackward(cudnn_handle, " << algorithm << ", "
       << "CUDNN_SOFTMAX_MODE_INSTANCE, "
       << "&alpha, "
       << "input_tensor_desc, " // y
       << "input1, "
       << "input_tensor_desc, " // dy
       << "input0, "
       << "&beta, "
       << "output_tensor_desc, " // dx
       << "output0));\n";
    lu << "CUDNN_SAFE_CALL(cudnnDestroyTensorDescriptor(input_tensor_desc));\n";
    lu << "CUDNN_SAFE_CALL(cudnnDestroyTensorDescriptor(output_tensor_desc));\n";

    return _lu;
}

LanguageUnit_p cuda::SoftmaxGrad::emit_dependency()
{
    LanguageUnit_p _lu(new LanguageUnit(get_function_name() + "_dep"));
    _lu->require(header::cuda);
    _lu->require(header::cudnn);
    //_lu->require(declaration::cudnn_handle);
    _lu->require(macro::CUDNN_SAFE_CALL);
    return _lu;
}

LanguageUnit_p cuda::SoftmaxGrad::emit_function_signature()
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

REGISTER_KERNEL_EMITTER(
    "SoftmaxGrad",                                                                 // op_name
    Device(CUDA_GPU).TypeConstraint(element::f32).Tag("cudnn_kernel").Priority(2), // attrs
    cuda::SoftmaxGrad)

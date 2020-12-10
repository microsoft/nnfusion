// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "max_pool.hpp"
#include "../cuda_cudnn.hpp"

using namespace nnfusion;
using namespace nnfusion::kernels;

cuda::MaxPool1D::MaxPool1D(shared_ptr<KernelContext> ctx)
    : BlockCudaEmitter(ctx)
{
    auto max_pool = static_pointer_cast<nnfusion::op::MaxPool>(ctx->gnode->get_op_ptr());
    input_shape = nnfusion::Shape(ctx->inputs[0]->get_shape());
    output_shape = nnfusion::Shape(ctx->outputs[0]->get_shape());
    window_shape = nnfusion::Shape(max_pool->get_window_shape());
    padding_below = nnfusion::Shape(max_pool->get_padding_below());
    padding_above = nnfusion::Shape(max_pool->get_padding_above());
    window_stride = nnfusion::Strides(max_pool->get_window_movement_strides());

    window_width = window_shape.back();
    window_stride_width = window_stride.back();
    input_width = input_shape.back();
    output_width = output_shape.back();

    input_type = ctx->inputs[0]->get_element_type();
    output_type = ctx->outputs[0]->get_element_type();

    // NNFUSION_CHECK(input_shape.size() == 3)
    //     << "Input shape size of MaxPool1D is invalid, shape size: " << input_shape.size()
    //     << "expected 3";

    std::stringstream tag;
    tag << "cuda_maxpool_" << input_type.c_type_string() << "_" << output_type.c_type_string()
        << "_iw" << std::to_string(input_width) << "_ow" << std::to_string(output_width) << "_ww"
        << std::to_string(window_width) << "_wst" << std::to_string(window_stride_width);
    custom_tag = tag.str();
}

LanguageUnit_p cuda::MaxPool1D::emit_function_body()
{
    if (input_shape.size() != 3)
        return nullptr;

    LanguageUnit_p _lu(new LanguageUnit(get_function_name()));
    auto& lu = *_lu;

    // Index into output tensor.
    lu << "size_t tid = blockIdx.x * blockDim.x + threadIdx.x;\n";
    lu << "if (tid < " << static_cast<uint32_t>(shape_size(output_shape)) << ")\n";
    lu.block_begin();
    {
        // Index into input tensor.
        lu << "size_t start = (tid / " << output_width << ") * " << input_width << " + "
           << " (tid % " << output_width << ") * " << window_stride[0] << ";\n";
        lu << input_type.c_type_string() << " max_val = " << TypeInfo::Get(input_type)->lowest()
           << ";\n";
        lu << "for (size_t i = start; i < start + " << window_width << "; i++)\n";
        lu.block_begin();
        {
            lu << "const " << input_type.c_type_string() << " input = input0[i];\n";
            lu << "if (input > max_val)\n";
            lu.block_begin();
            {
                lu << "max_val = input;\n";
            }
            lu.block_end();
        }
        lu.block_end();
        lu << "output0[tid] = max_val;\n";
    }
    lu.block_end();

    return _lu;
}

void cuda::MaxPool1D::set_launch_config()
{
    size_t nthreads = shape_size(output_shape);
    // TODO: currently we set it to 64, will add tuning method later.
    uint32_t block_size_x = 64;
    uint32_t aligned_grid_size_x =
        align_to_block_size(static_cast<uint32_t>(nthreads), block_size_x);

    m_gridDim = dim3(aligned_grid_size_x, 1, 1);
    m_blockDim = dim3(block_size_x, 1, 1);
}

LanguageUnit_p cuda::MaxPool1D::emit_dependency()
{
    LanguageUnit_p _lu(new LanguageUnit(get_function_name() + "_dep"));

    _lu->require(header::cuda);

    return _lu;
}

cuda::MaxPoolmD::MaxPoolmD(shared_ptr<KernelContext> ctx)
    : CudaLibEmitter(ctx)
{
    auto max_pool = static_pointer_cast<nnfusion::op::MaxPool>(ctx->gnode->get_op_ptr());
    input_type = ctx->inputs[0]->get_element_type();
    output_type = ctx->outputs[0]->get_element_type();
    input_shape = nnfusion::Shape(ctx->inputs[0]->get_shape());
    output_shape = nnfusion::Shape(ctx->outputs[0]->get_shape());
    window_shape = nnfusion::Shape(max_pool->get_window_shape());
    padding_below = nnfusion::Shape(max_pool->get_padding_below());
    padding_above = nnfusion::Shape(max_pool->get_padding_above());
    window_stride = nnfusion::Strides(max_pool->get_window_movement_strides());

    std::stringstream tag;
    tag << "cudnn_maxpool_dtype_" << output_type.c_type_string() << "_i" << join(input_shape, "_")
        << "_o" << join(output_shape, "_") << "_ws" << join(window_shape, "_") << "_wst"
        << join(window_stride, "_") << "_pb" << join(padding_below, "_") << "_pb"
        << join(padding_above, "_");
    custom_tag = tag.str();
}

LanguageUnit_p cuda::MaxPoolmD::emit_function_body()
{
    if (input_shape.size() != 4 && input_shape.size() != 5)
        return nullptr;

    LanguageUnit_p _lu(new LanguageUnit(get_function_name()));
    auto& lu = *_lu;

    auto input_desc = cudnn_tensor_descriptor_from_shape(input_shape, "input_desc", input_type);
    auto output_desc = cudnn_tensor_descriptor_from_shape(output_shape, "output_desc", output_type);
    lu << input_desc->get_code();
    lu << output_desc->get_code();

    lu << "cudnnPoolingDescriptor_t desc;\n";
    lu << "cudnnCreatePoolingDescriptor(&desc);\n";
    if (input_shape.size() == 4)
    {
        lu << "CUDNN_SAFE_CALL(cudnnSetPooling2dDescriptor(desc,"
           << " CUDNN_POOLING_MAX,"
           << " CUDNN_NOT_PROPAGATE_NAN," << static_cast<int>(window_shape[0]) << ", "
           << static_cast<int>(window_shape[1]) << ", " << static_cast<int>(padding_below[0])
           << ", " << static_cast<int>(padding_below[1]) << ", "
           << static_cast<int>(window_stride[0]) << ", " << static_cast<int>(window_stride[1])
           << "));\n";
    }
    else /*op->input_shape.size() == 5*/
    {
        std::vector<int> w_strides(window_stride.size());
        std::vector<int> w_shape(window_shape.size());
        std::vector<int> w_padding(padding_below.size());
        for (int i = 0; i < window_shape.size(); i++)
        {
            w_shape[i] = static_cast<int>(window_shape[i]);
            w_strides[i] = static_cast<int>(window_stride[i]);
            w_padding[i] = static_cast<int>(padding_below[i]);
        }

        auto expand_vector_int = [](string name, vector<int>& d) {
            stringstream ss;
            NNFUSION_CHECK(d.size() > 0);
            ss << "int " << name << "[] = {";
            for (int i = 0; i + 1 < d.size(); i++)
                ss << to_string(d[i]) << ", ";
            ss << to_string(d.back()) << "}\n";
            return ss.str();
        };

        lu << expand_vector_int("w_shape", w_shape);
        lu << expand_vector_int("w_strides", w_strides);
        lu << expand_vector_int("w_padding", w_padding);

        lu << "CUDNN_SAFE_CALL(cudnnSetPoolingNdDescriptor(desc, "
           << "CUDNN_POOLING_MAX, "
           << "CUDNN_NOT_PROPAGATE_NAN, "
           << "3, w_shape, w_padding, w_strides));\n";
    }

    lu << "const float alpha = 1.0;\n";
    lu << "const float beta = 0.0;\n";

    lu << "CUDNN_SAFE_CALL(cudnnPoolingForward(cudnn_handle,"
       << " desc,"
       << " &alpha,"
       << " input_desc,"
       << " input0,"
       << " &beta,"
       << " output_desc,"
       << " output0));\n";

    lu << "CUDNN_SAFE_CALL(cudnnDestroyTensorDescriptor(input_desc));\n";
    lu << "CUDNN_SAFE_CALL(cudnnDestroyTensorDescriptor(output_desc));\n";
    lu << "CUDNN_SAFE_CALL(cudnnDestroyPoolingDescriptor(desc));\n";

    return _lu;
}

LanguageUnit_p cuda::MaxPoolmD::emit_dependency()
{
    LanguageUnit_p _lu(new LanguageUnit(get_function_name() + "_dep"));

    _lu->require(header::cudnn);
    //_lu->require(declaration::cudnn_handle);
    _lu->require(macro::CUDNN_SAFE_CALL);

    return _lu;
}

LanguageUnit_p cuda::MaxPoolmD::emit_function_signature()
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
    "MaxPool",                                                                    // op_name
    Device(CUDA_GPU).TypeConstraint(element::f32).Tag("cuda_kernel").Priority(2), // attrs
    cuda::MaxPool1D)                                                              // constructor

REGISTER_KERNEL_EMITTER(
    "MaxPool",                                                                     // op_name
    Device(CUDA_GPU).TypeConstraint(element::f32).Tag("cudnn_kernel").Priority(2), // attrs
    cuda::MaxPoolmD)                                                               // constructor

// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "batch_norm.hpp"

using namespace nnfusion;
using namespace nnfusion::kernels;

cuda::BatchNorm::BatchNorm(shared_ptr<KernelContext> ctx)
    : CudaLibEmitter(ctx)
{
    // nnfusion::op::BatchNormInferece <-> nnfusion::ir::BatchNorm
    auto bn_op = static_pointer_cast<nnfusion::op::BatchNormInference>(ctx->gnode->get_op_ptr());
    dtype = nnfusion::element::Type(ctx->outputs[0]->get_element_type());
    // <todo> need to check the index
    tensor_shape = nnfusion::Shape(ctx->inputs[2]->get_shape());
    param_shape = nnfusion::Shape(ctx->inputs[0]->get_shape());
    epsilon = bn_op->get_eps_value();

    std::stringstream tag;
    tag << "cudnn_batch_norm"
        << "_dtype_" << dtype.c_type_string() << "_i_" << join(tensor_shape, "_") << "_i_"
        << join(param_shape, "_") << "_" << ctx->outputs[0]->get_name();
    custom_tag = tag.str();
}

LanguageUnit_p cuda::BatchNorm::emit_function_body()
{
    LanguageUnit_p _lu(new LanguageUnit(get_function_name()));
    auto& lu = *_lu;
    auto tensor_desc = cudnn_tensor_descriptor_from_shape(tensor_shape, "tensor_desc", dtype);
    lu << tensor_desc->get_code();
    // derived_param_desc
    lu << "cudnnTensorDescriptor_t derived_param_desc;\n";
    lu << "CUDNN_SAFE_CALL(cudnnCreateTensorDescriptor(&derived_param_desc));\n";
    lu << "CUDNN_SAFE_CALL(cudnnDeriveBNTensorDescriptor(derived_param_desc, tensor_desc, "
          "CUDNN_BATCHNORM_SPATIAL));\n";
    lu << "const float alpha = 1.0;\n";
    lu << "const float beta = 0.0;\n";
    lu << "CUDNN_SAFE_CALL(cudnnBatchNormalizationForwardInference(cudnn_handle,"
       << " CUDNN_BATCHNORM_SPATIAL,"
       << " &alpha,"
       << " &beta,"
       << " tensor_desc,"
       << " input2,"
       << " tensor_desc,"
       << " output0,"
       << " derived_param_desc,"
       << " input0," // gain
       << " input1," // bias
       << " input3," // mean
       << " input4," // variance
       << epsilon << "));\n";

    lu << "CUDNN_SAFE_CALL(cudnnDestroyTensorDescriptor(tensor_desc));\n";
    lu << "CUDNN_SAFE_CALL(cudnnDestroyTensorDescriptor(derived_param_desc));\n";
    return _lu;
}

LanguageUnit_p cuda::BatchNorm::emit_dependency()
{
    LanguageUnit_p _lu(new LanguageUnit(get_function_name() + "_dep"));
    _lu->require(header::cuda);
    _lu->require(header::cudnn);
    _lu->require(header::stdexcept);
    _lu->require(header::sstream);
    _lu->require(macro::CUDNN_SAFE_CALL);
    //_lu->require(declaration::cudnn_handle);
    return _lu;
}

LanguageUnit_p cuda::BatchNorm::emit_function_signature()
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

cuda::BatchNormNCHW::BatchNormNCHW(shared_ptr<KernelContext> ctx)
    : BlockCudaEmitter(ctx)
{
    // nnfusion::op::BatchNormInferece <-> nnfusion::ir::BatchNormNCHW
    auto bn_op = static_pointer_cast<nnfusion::op::BatchNormInference>(ctx->gnode->get_op_ptr());
    dtype = nnfusion::element::Type(ctx->outputs[0]->get_element_type());
    // <todo> need to check the index
    tensor_shape = nnfusion::Shape(ctx->inputs[2]->get_shape());
    param_shape = nnfusion::Shape(ctx->inputs[0]->get_shape());
    epsilon = bn_op->get_eps_value();

    // CHECK(tensor_shape.size() == 4) << "BatchNormNCHW: input must be 4-dimensional";
    // CHECK(param_shape.size() == 1) << "BatchNormNCHW: param must be 1-dimensional";
    // CHECK(tensor_shape[1] == param_shape[0])
    //     << "BatchNormNCHW: channels of input and param do not match";

    std::stringstream tag;
    tag << "cuda_batch_norm_nchw"
        << "_dtype_" << dtype.c_type_string() << "_i_" << join(tensor_shape, "_") << "_i_"
        << join(param_shape, "_") << "_" << ctx->outputs[0]->get_name();
    custom_tag = tag.str();
}

LanguageUnit_p cuda::BatchNormNCHW::emit_function_body()
{
    if (tensor_shape.size() != 4)
    {
        return nullptr;
    }
    if (param_shape.size() != 1)
    {
        return nullptr;
    }
    if (tensor_shape[1] != param_shape[0])
    {
        return nullptr;
    }

    LanguageUnit_p _lu(new LanguageUnit(get_function_name()));
    auto& lu = *_lu;

    const float alpha = 1.0;
    const float beta = 0.0;
    const size_t batch_size = tensor_shape[0];
    const size_t channel_size = tensor_shape[1];
    const size_t height_size = tensor_shape[2];
    const size_t width_size = tensor_shape[3];

    lu << "const int st = blockIdx.x * " << height_size << " * " << width_size << ";\n";
    lu << "const int c_id = blockIdx.x % " << channel_size << ";\n";
    lu << "#pragma unroll 1\n";
    lu << "for (int i = threadIdx.x; i < " << height_size << " * " << width_size
       << "; i += blockDim.x)\n";
    lu.block_begin();
    lu << "output0[st + i] = ";
    if (beta != 0)
    {
        lu << beta << " * output0[st + i] + ";
    }
    if (alpha != 1)
    {
        lu << alpha << " * ";
    }
    lu << "(input1[c_id] + (input0[c_id] * "
          "(input2[st + i] - input3[c_id]) / sqrtf("
       << epsilon << " + input4[c_id])));\n";
    // lu << "output0[st + i] = " << beta << " * output0[st + i] + " << alpha
    //    << " * (input1[c_id] + (input0[c_id] * "
    //       "(input2[st + i] - input3[c_id]) / sqrtf("
    //    << epsilon << " + input4[c_id])));\n";
    lu.block_end();

    return _lu;
}

LanguageUnit_p cuda::BatchNormNCHW::emit_dependency()
{
    LanguageUnit_p _lu(new LanguageUnit(get_function_name() + "_dep"));
    _lu->require(header::cuda);
    return _lu;
}

void cuda::BatchNormNCHW::set_launch_config()
{
    const size_t batch_size = tensor_shape[0];
    const size_t channel_size = tensor_shape[1];
    const size_t height_size = tensor_shape[2];
    const size_t width_size = tensor_shape[3];

    m_gridDim = dim3(batch_size * channel_size, 1, 1);
    m_blockDim = dim3(std::min(512, (int)(height_size * width_size)), 1, 1);
}

using namespace nnfusion;
using namespace nnfusion::kernels;

REGISTER_KERNEL_EMITTER(
    "BatchNormInference",                                                   // op_name
    Device(CUDA_GPU).TypeConstraint(element::f32).Tag("cudnn").Priority(2), // attrs
    cuda::BatchNorm)                                                        // constructor
REGISTER_KERNEL_EMITTER(
    "BatchNormInference",                                                  // op_name
    Device(CUDA_GPU).TypeConstraint(element::f32).Tag("cuda").Priority(2), // attrs
    cuda::BatchNormNCHW)                                                   // constructor
REGISTER_KERNEL_EMITTER(
    "BatchNormInference",                                                  // op_name
    Device(ROCM_GPU).TypeConstraint(element::f32).Tag("cuda").Priority(2), // attrs
    cuda::BatchNormNCHW)                                                   // constructor

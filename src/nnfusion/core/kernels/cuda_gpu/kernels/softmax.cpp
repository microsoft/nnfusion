// Microsoft (c) 2019, NNFusion Team

#include "softmax.hpp"
#include "nnfusion/core/operators/generic_op/generic_op.hpp"

using namespace nnfusion;
using namespace nnfusion::kernels;

cuda::Softmax::Softmax(shared_ptr<KernelContext> ctx)
    : CudaLibEmitter(ctx)
{
    auto node = static_pointer_cast<nnfusion::op::Softmax>(ctx->gnode->get_op_ptr());
    input_shape = nnfusion::Shape(ctx->inputs[0]->get_shape());
    output_shape = nnfusion::Shape(ctx->outputs[0]->get_shape());
}

LanguageUnit_p
    cuda::Softmax::cudnn_tensor_descriptor_from_shape_for_softmax(const nnfusion::Shape& shape,
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

LanguageUnit_p cuda::Softmax::emit_function_body()
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
    lu << "CUDNN_SAFE_CALL(cudnnSetStream(global_cudnn_handle, stream));\n";
    lu << input_tensor_desc->get_code();
    lu << output_tensor_desc->get_code();
    lu << "const float alpha = 1.0;\n";
    lu << "const float beta = 0.0;\n";
    lu << "CUDNN_SAFE_CALL(cudnnSoftmaxForward(global_cudnn_handle, "
       << "CUDNN_SOFTMAX_ACCURATE, "
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
    _lu->require(declaration::global_cudnn_handle);
    _lu->require(macro::CUDNN_SAFE_CALL);
    return _lu;
}

REGISTER_KERNEL_EMITTER("Softmax",                                                     // op_name
                        Device(CUDA_GPU).TypeConstraint(DT_FLOAT).Tag("cudnn_kernel"), // attrs
                        cuda::Softmax) // constructor

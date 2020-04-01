// Microsoft (c) 2019, NNFusion Team

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
    auto tensor_desc = cudnn_tensor_descriptor_from_shape(tensor_shape, "tensor_desc");
    lu << "CUDNN_SAFE_CALL(cudnnSetStream(global_cudnn_handle, stream));\n";
    lu << tensor_desc->get_code();
    // derived_param_desc
    lu << "cudnnTensorDescriptor_t derived_param_desc;\n";
    lu << "CUDNN_SAFE_CALL(cudnnCreateTensorDescriptor(&derived_param_desc));\n";
    lu << "CUDNN_SAFE_CALL(cudnnDeriveBNTensorDescriptor(derived_param_desc, tensor_desc, "
          "CUDNN_BATCHNORM_SPATIAL));\n";
    lu << "const float alpha = 1.0;\n";
    lu << "const float beta = 0.0;\n";
    lu << "CUDNN_SAFE_CALL(cudnnBatchNormalizationForwardInference(global_cudnn_handle,"
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
    _lu->require(macro::CUBLAS_SAFE_CALL);
    _lu->require(declaration::global_cublas_handle);
    return _lu;
}

using namespace nnfusion;
using namespace nnfusion::kernels;

REGISTER_KERNEL_EMITTER("BatchNormInference",                                   // op_name
                        Device(CUDA_GPU).TypeConstraint(DT_FLOAT).Tag("cudnn"), // attrs
                        cuda::BatchNorm)                                        // constructor
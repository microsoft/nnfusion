// Microsoft (c) 2019, NNFusion Team

#include "convolution.hpp"
#include "../cuda_cudnn.hpp"

using namespace nnfusion;
using namespace nnfusion::kernels;

cuda::ConvolutionCudnn::ConvolutionCudnn(shared_ptr<KernelContext> ctx)
    : CudaLibEmitter(ctx)
{
    auto conv = static_pointer_cast<nnfusion::op::Convolution>(ctx->gnode->get_op_ptr());

    input_shape = ctx->inputs[0]->get_shape();
    filter_shape = ctx->inputs[1]->get_shape();
    output_shape = ctx->outputs[0]->get_shape();
    window_dilation_strides = conv->get_window_dilation_strides();
    window_movement_strides = conv->get_window_movement_strides();
    data_dilation_strides = conv->get_data_dilation_strides();
    padding_below_diff = conv->get_padding_below();
    padding_above_diff = conv->get_padding_above();
    dtype = ctx->outputs[0]->get_element_type().c_type_string();

    std::stringstream tag;
    tag << "cudnn_convolution_op_" << dtype << "_i" << join(input_shape, "_") << "_w"
        << join(filter_shape, "_") << "_o" << join(output_shape, "_") << "_ws"
        << join(window_movement_strides, "_") << "_wd" << join(window_dilation_strides, "_") << "_p"
        << join(padding_below_diff, "_");
    custom_tag = tag.str();
}

LanguageUnit_p cuda::ConvolutionCudnn::emit_function_body()
{
    // <TODO> full feature of cudnn convolutoin
    bool is_deconvolution = false;
    for (auto a : data_dilation_strides)
    {
        if (a != 1)
        {
            is_deconvolution = true;
            break;
        }
    }
    if (is_deconvolution)
    {
        LOG(WARNING) << "Deconvolution is not supported by now.";
        return nullptr;
    }

    if (padding_below_diff != padding_above_diff)
    {
        LOG(WARNING) << "Asymetric padding is not supported by now.";
        return nullptr;
    }

    // emit code
    LanguageUnit_p _lu(new LanguageUnit(get_function_name()));
    auto& lu = *_lu;
    lu << "CUDNN_SAFE_CALL(cudnnSetStream(global_cudnn_handle, stream));\n";

    Shape padding_below(padding_below_diff.size(), 0);

    for (int i = 0; i < padding_below.size(); i++)
    {
        padding_below[i] = static_cast<size_t>(padding_below_diff[i]);
    }

    {
        // lu << "cudnnDataType_t data_type = " << get_cudnn_datatype(dtype) << ";\n";
        lu << cudnn_tensor_descriptor_from_shape(input_shape, "tensor_desc_0")->get_code();
        lu << cudnn_tensor_descriptor_from_shape(output_shape, "tensor_desc_1")->get_code();
        lu << get_cudnn_filter_descriptor(filter_shape, "filter_desc")->get_code();
        lu << get_cudnn_convolution_descriptor(
                  padding_below, window_movement_strides, window_dilation_strides, "conv_desc")
                  ->get_code();

        lu << R"(
static bool selected_algo = false;
static cudnnConvolutionFwdAlgo_t conv_fwd_algo = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM;

if (!selected_algo) {
    int num_algos;
    int max_algos = 0;
    // cudnnGetConvolutionForwardAlgorithm_v7;
    CUDNN_SAFE_CALL(
        cudnnGetConvolutionForwardAlgorithmMaxCount(global_cudnn_handle, &max_algos));
    std::vector<cudnnConvolutionFwdAlgoPerf_t> results(max_algos);
    CUDNN_SAFE_CALL(cudnnFindConvolutionForwardAlgorithm(global_cudnn_handle,
                                            tensor_desc_0,
                                            filter_desc,
                                            conv_desc,
                                            tensor_desc_1,
                                            static_cast<int>(results.size()),
                                            &num_algos,
                                            results.data()));
    results.resize(num_algos);
    for (size_t i = 0; i != results.size(); ++i) {
        cudnnConvolutionFwdAlgoPerf_t const& result = results[i];
        if (result.status == CUDNN_STATUS_SUCCESS) {
            conv_fwd_algo = result.algo;
            break;
        }
    }
    selected_algo = true;
})";
        lu << "\n";
        lu << "const float alpha = 1.0;\n";
        lu << "const float beta = 0.0;\n";
        lu << "static void *workspace_ptr = NULL;\n"
           << "static size_t workspace_size_in_bytes = 0;\n";
        lu << "if (!workspace_ptr)\n";
        lu.block_begin();
        lu << "CUDNN_SAFE_CALL(cudnnGetConvolutionForwardWorkspaceSize(global_cudnn_handle, "
           << "tensor_desc_0, "
           << "filter_desc, "
           << "conv_desc, "
           << "tensor_desc_1, "
           << "conv_fwd_algo, "
           << "&workspace_size_in_bytes));\n";
        //lu << "void *workspace_ptr;\n"
        lu << "CUDA_SAFE_CALL(cudaMalloc(&workspace_ptr, workspace_size_in_bytes));\n";
        lu.block_end();
        lu << "CUDNN_SAFE_CALL(cudnnConvolutionForward(global_cudnn_handle, "
           << "&alpha, "
           << "tensor_desc_0, "
           << "input0,"
           << "filter_desc, "
           << "input1, "
           << "conv_desc, "
           << "conv_fwd_algo, "
           << "workspace_ptr, "
           << "workspace_size_in_bytes, "
           << "&beta, "
           << "tensor_desc_1, "
           << "output0));\n";
        //lu << "CUDA_SAFE_CALL(cudaFree(workspace_ptr));\n";
        lu << "CUDNN_SAFE_CALL(cudnnDestroyTensorDescriptor(tensor_desc_0));\n";
        lu << "CUDNN_SAFE_CALL(cudnnDestroyTensorDescriptor(tensor_desc_1));\n";
        lu << "CUDNN_SAFE_CALL(cudnnDestroyFilterDescriptor(filter_desc));\n";
        lu << "CUDNN_SAFE_CALL(cudnnDestroyConvolutionDescriptor(conv_desc));\n";
    }

    return _lu;
}

LanguageUnit_p cuda::ConvolutionCudnn::emit_dependency()
{
    LanguageUnit_p _lu(new LanguageUnit(get_function_name() + "_dep"));

    _lu->require(header::cudnn);
    _lu->require(declaration::global_cudnn_handle);
    _lu->require(macro::CUDNN_SAFE_CALL);
    _lu->require(header::vector);

    return _lu;
}

REGISTER_KERNEL_EMITTER("Convolution",                                                 // op_name
                        Device(CUDA_GPU).TypeConstraint(DT_FLOAT).Tag("cudnn_kernel"), // attrs
                        cuda::ConvolutionCudnn) // constructor

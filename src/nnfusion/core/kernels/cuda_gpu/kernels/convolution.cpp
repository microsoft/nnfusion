// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "convolution.hpp"
#include "../cuda_cudnn.hpp"

using namespace nnfusion;
using namespace nnfusion::kernels;

cuda::ConvolutionCudnn::ConvolutionCudnn(shared_ptr<KernelContext> ctx)
    : CudaLibEmitter(ctx)
{
    auto conv = static_pointer_cast<nnfusion::op::Convolution>(ctx->gnode->get_op_ptr());

    input_type = ctx->inputs[0]->get_element_type();
    filter_type = ctx->inputs[1]->get_element_type();
    output_type = ctx->outputs[0]->get_element_type();
    NNFUSION_CHECK(input_type == filter_type && input_type == output_type)
        << "Convolution input datatype (" << input_type
        << ") should be the same with that of filter (" << filter_type << "), and that of output ("
        << output_type << ").";
    conv_type = input_type;
    input_shape = ctx->inputs[0]->get_shape();
    filter_shape = ctx->inputs[1]->get_shape();
    output_shape = ctx->outputs[0]->get_shape();
    window_dilation_strides = conv->get_window_dilation_strides();
    window_movement_strides = conv->get_window_movement_strides();
    data_dilation_strides = conv->get_data_dilation_strides();
    padding_below_diff = conv->get_padding_below();
    padding_above_diff = conv->get_padding_above();
    data_format = conv->get_data_format();
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
        NNFUSION_LOG(NNFUSION_WARNING) << "Deconvolution is not supported by now.";
        return nullptr;
    }

    if (padding_below_diff != padding_above_diff)
    {
        NNFUSION_LOG(NNFUSION_WARNING) << "Asymetric padding is not supported by now.";
        return nullptr;
    }

    // Conv1D: convert Conv1D to Conv2D
    if (data_format == "NCW")
    {
        input_shape = {input_shape[0], input_shape[1], 1, input_shape[2]};
        filter_shape = {filter_shape[0], filter_shape[1], 1, filter_shape[2]};
        output_shape = {output_shape[0], output_shape[1], 1, output_shape[2]};
        window_dilation_strides = {1, window_dilation_strides[0]};
        window_movement_strides = {1, window_movement_strides[0]};
        data_dilation_strides = {1, data_dilation_strides[0]};
        padding_below_diff = {0, padding_below_diff[0]};
        padding_above_diff = {0, padding_above_diff[0]};
    }

    // emit code
    LanguageUnit_p _lu(new LanguageUnit(get_function_name()));
    auto& lu = *_lu;

    Shape padding_below(padding_below_diff.size(), 0);

    for (int i = 0; i < padding_below.size(); i++)
    {
        padding_below[i] = static_cast<size_t>(padding_below_diff[i]);
    }

    {
        // lu << "cudnnDataType_t data_type = " << get_cudnn_datatype(dtype) << ";\n";
        lu << cudnn_tensor_descriptor_from_shape(input_shape, "tensor_desc_0", input_type)
                  ->get_code();
        lu << cudnn_tensor_descriptor_from_shape(output_shape, "tensor_desc_1", output_type)
                  ->get_code();
        lu << get_cudnn_filter_descriptor(filter_shape, "filter_desc", filter_type)->get_code();
        lu << get_cudnn_convolution_descriptor(padding_below,
                                               window_movement_strides,
                                               window_dilation_strides,
                                               "conv_desc",
                                               conv_type)
                  ->get_code();

        lu << R"(
static bool selected_algo = false;
static cudnnConvolutionFwdAlgo_t conv_fwd_algo = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM;

if (!selected_algo) {
    int num_algos;
    int max_algos = 0;
    // cudnnGetConvolutionForwardAlgorithm_v7;
    CUDNN_SAFE_CALL(
        cudnnGetConvolutionForwardAlgorithmMaxCount(cudnn_handle, &max_algos));
    std::vector<cudnnConvolutionFwdAlgoPerf_t> results(max_algos);
    CUDNN_SAFE_CALL(cudnnFindConvolutionForwardAlgorithm(cudnn_handle,
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
        lu << "CUDNN_SAFE_CALL(cudnnGetConvolutionForwardWorkspaceSize(cudnn_handle, "
           << "tensor_desc_0, "
           << "filter_desc, "
           << "conv_desc, "
           << "tensor_desc_1, "
           << "conv_fwd_algo, "
           << "&workspace_size_in_bytes));\n";
        //lu << "void *workspace_ptr;\n"
        lu << "CUDA_SAFE_CALL(cudaMalloc(&workspace_ptr, workspace_size_in_bytes));\n";
        lu.block_end();
        lu << "CUDNN_SAFE_CALL(cudnnConvolutionForward(cudnn_handle, "
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
    //_lu->require(declaration::cudnn_handle);
    _lu->require(macro::CUDNN_SAFE_CALL);
    _lu->require(header::vector);

    return _lu;
}

LanguageUnit_p cuda::ConvolutionCudnn::emit_function_signature()
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
    "Convolution",                                                                 // op_name
    Device(CUDA_GPU).TypeConstraint(element::f32).Tag("cudnn_kernel").Priority(2), // attrs
    cuda::ConvolutionCudnn)                                                        // constructor

REGISTER_KERNEL_EMITTER(
    "Convolution",                                                                 // op_name
    Device(CUDA_GPU).TypeConstraint(element::f16).Tag("cudnn_kernel").Priority(2), // attrs
    cuda::ConvolutionCudnn)                                                        // constructor

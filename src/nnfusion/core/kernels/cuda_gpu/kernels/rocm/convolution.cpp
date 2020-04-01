// Microsoft (c) 2019, NNFusion Team

#include "../../cuda_cudnn.hpp"
#include "../../cuda_emitter.hpp"
#include "../../cuda_langunit.hpp"

namespace nnfusion
{
    namespace kernels
    {
        namespace cuda
        {
            class RocmConvolutionCudnn : public CudaLibEmitter
            {
            public:
                RocmConvolutionCudnn(shared_ptr<KernelContext> ctx)
                    : CudaLibEmitter(ctx)
                {
                }

                LanguageUnit_p emit_function_body() override
                {
                    auto& ctx = m_context;

                    auto& input_shape = ctx->inputs[0]->get_shape();
                    auto& filter_shape = ctx->inputs[1]->get_shape();
                    auto& output_shape = ctx->outputs[0]->get_shape();

                    auto conv =
                        static_pointer_cast<nnfusion::op::Convolution>(ctx->gnode->get_op_ptr());
                    auto& window_dilation_strides = conv->get_window_dilation_strides();
                    auto& window_movement_strides = conv->get_window_movement_strides();
                    auto& data_dilation_strides = conv->get_data_dilation_strides();
                    auto& padding_below_diff = conv->get_padding_below();
                    auto& padding_above_diff = conv->get_padding_above();
                    auto& dtype = ctx->outputs[0]->get_element_type().c_type_string();

                    if (dtype != "float")
                        return nullptr;

                    std::stringstream tag;
                    tag << "cudnn_convolution_op_" << dtype << "_i" << join(input_shape, "_")
                        << "_w" << join(filter_shape, "_") << "_o" << join(output_shape, "_")
                        << "_ws" << join(window_movement_strides, "_") << "_wd"
                        << join(window_dilation_strides, "_") << "_p"
                        << join(padding_below_diff, "_");
                    custom_tag = tag.str();

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

                    Shape padding_below(padding_below_diff.size(), 0);

                    for (int i = 0; i < padding_below.size(); i++)
                    {
                        padding_below[i] = static_cast<size_t>(padding_below_diff[i]);
                    }

                    {
                        lu << "CUDNN_SAFE_CALL(cudnnSetStream(global_cudnn_handle, stream));\n";
                        // lu << "cudnnDataType_t data_type = " << get_cudnn_datatype(dtype) << ";\n";
                        lu << cudnn_tensor_descriptor_from_shape(input_shape, "tensor_desc_0")
                                  ->get_code();
                        lu << cudnn_tensor_descriptor_from_shape(output_shape, "tensor_desc_1")
                                  ->get_code();
                        lu << get_cudnn_filter_descriptor(filter_shape, "filter_desc")->get_code();
                        lu << get_cudnn_convolution_descriptor(padding_below,
                                                               window_movement_strides,
                                                               window_dilation_strides,
                                                               "conv_desc")
                                  ->get_code();

                        lu << R"(
	static bool inited = false; static int returnedAlgoCount, fastest = 0;
	static miopenConvAlgoPerf_t perfResults[4];
	static size_t workspace_size_in_bytes = 0;
	static void *workspace_ptr = NULL;
	const float alpha = 1.0f, beta = 0.0f;
	if (!inited) {
		inited = true;
		fprintf(stderr, "[MIOpen] Convolution running auto-tune for Forward-Data;\n");
		CUDNN_SAFE_CALL(miopenFindConvolutionForwardAlgorithm(global_cudnn_handle,
			tensor_desc_0, input0, filter_desc, input1, conv_desc, tensor_desc_1, output0,
			4, &returnedAlgoCount, perfResults, workspace_ptr, workspace_size_in_bytes, false));
		for (int i = 1; i < returnedAlgoCount; ++i)
			if (perfResults[i].time < perfResults[fastest].time)
				fastest = i;
		workspace_size_in_bytes = perfResults[fastest].memory;
		CUDA_SAFE_CALL(cudaMalloc(&workspace_ptr, workspace_size_in_bytes));
	}
	CUDNN_SAFE_CALL(miopenConvolutionForward(global_cudnn_handle,
		&alpha, tensor_desc_0, input0, filter_desc, input1, conv_desc, perfResults[fastest].fwd_algo,
		&beta, tensor_desc_1, output0, workspace_ptr, workspace_size_in_bytes));
	CUDNN_SAFE_CALL(cudnnDestroyTensorDescriptor(tensor_desc_0));
	CUDNN_SAFE_CALL(cudnnDestroyTensorDescriptor(tensor_desc_1));
	CUDNN_SAFE_CALL(cudnnDestroyFilterDescriptor(filter_desc));
	CUDNN_SAFE_CALL(cudnnDestroyConvolutionDescriptor(conv_desc));
)";
                    }

                    return _lu;
                }

                LanguageUnit_p emit_dependency() override
                {
                    LanguageUnit_p _lu(new LanguageUnit(get_function_name() + "_dep"));

                    _lu->require(header::cudnn);
                    _lu->require(declaration::global_cudnn_handle);
                    _lu->require(macro::CUDNN_SAFE_CALL);

                    return _lu;
                }
            };
        } // namespace cuda

    } // namespace kernels
} // namespace nnfusion

using namespace nnfusion;
using namespace nnfusion::kernels;

REGISTER_KERNEL_EMITTER("Convolution",                                                 // op_name
                        Device(ROCM_GPU).TypeConstraint(DT_FLOAT).Tag("cudnn_kernel"), // attrs
                        cuda::RocmConvolutionCudnn) // constructor

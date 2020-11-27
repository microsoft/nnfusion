// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "../../cuda_cudnn.hpp"
#include "../../cuda_emitter.hpp"
#include "../../cuda_langunit.hpp"

#include "nnfusion/core/operators/generic_op/generic_op.hpp"

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

                    auto input_shape = ctx->inputs[0]->get_shape();
                    auto filter_shape = ctx->inputs[1]->get_shape();
                    auto output_shape = ctx->outputs[0]->get_shape();

                    auto conv =
                        static_pointer_cast<nnfusion::op::Convolution>(ctx->gnode->get_op_ptr());
                    auto window_dilation_strides = conv->get_window_dilation_strides();
                    auto window_movement_strides = conv->get_window_movement_strides();
                    auto data_dilation_strides = conv->get_data_dilation_strides();
                    auto padding_below_diff = conv->get_padding_below();
                    auto padding_above_diff = conv->get_padding_above();
                    auto data_format = conv->get_data_format();
                    auto dtype = ctx->outputs[0]->get_element_type().c_type_string();

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
                        NNFUSION_LOG(NNFUSION_WARNING) << "Deconvolution is not supported by now.";
                        return nullptr;
                    }

                    if (padding_below_diff != padding_above_diff)
                    {
                        NNFUSION_LOG(NNFUSION_WARNING)
                            << "Asymetric padding is not supported by now.";
                        return nullptr;
                    }

                    if (!(data_format == "NCW" || data_format == "NCHW"))
                    {
                        NNFUSION_LOG(NNFUSION_WARNING) << "Convolution with " << data_format
                                                       << " format is not supported by now.";
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
                        std::map<std::string, size_t> j_map;
                        j_map["N"] = input_shape[0];
                        j_map["CI"] = input_shape[1];
                        j_map["H"] = input_shape[2];
                        j_map["W"] = input_shape[3];
                        j_map["CO"] = output_shape[1];
                        j_map["HO"] = output_shape[2];
                        j_map["WO"] = output_shape[3];
                        j_map["KH"] = filter_shape[2];
                        j_map["KW"] = filter_shape[3];
                        j_map["S0"] = window_movement_strides[0];
                        j_map["S1"] = window_movement_strides[1];
                        j_map["P0"] = padding_below[0];
                        j_map["P1"] = padding_below[1];

                        auto str = op::create_code_from_template(R"(
    miopenTensorDescriptor_t tensor_desc_0, tensor_desc_1, filter_desc;
    miopenConvolutionDescriptor_t conv_desc;

    CUDNN_SAFE_CALL(miopenCreateTensorDescriptor(&tensor_desc_0));
    CUDNN_SAFE_CALL(miopenCreateTensorDescriptor(&tensor_desc_1));
    CUDNN_SAFE_CALL(miopenCreateTensorDescriptor(&filter_desc));
    CUDNN_SAFE_CALL(miopenCreateConvolutionDescriptor(&conv_desc));

    CUDNN_SAFE_CALL(miopenSet4dTensorDescriptor(tensor_desc_0, CUDNN_DATA_FLOAT, @N@, @CI@, @H@, @W@));
    CUDNN_SAFE_CALL(miopenSet4dTensorDescriptor(tensor_desc_1, CUDNN_DATA_FLOAT, @N@, @CO@, @HO@, @WO@));
    CUDNN_SAFE_CALL(miopenSet4dTensorDescriptor(filter_desc, CUDNN_DATA_FLOAT, @CO@, @CI@, @KH@, @KW@));
    CUDNN_SAFE_CALL(miopenInitConvolutionDescriptor(conv_desc, CUDNN_CROSS_CORRELATION, @P0@, @P1@, @S0@, @S1@, 1, 1));

	static bool inited = false; static int returnedAlgoCount, fastest = 0;
	static miopenConvAlgoPerf_t perfResults[4];
	static size_t workspace_size_in_bytes = 0;
	static void *workspace_ptr = NULL;
	const float alpha = 1.0f, beta = 0.0f;
	if (!inited) {
		inited = true;
		fprintf(stderr, "[MIOpen] Convolution running auto-tune for Forward-Data;\n");
        try {
		CUDNN_SAFE_CALL(miopenFindConvolutionForwardAlgorithm(cudnn_handle,
			tensor_desc_0, input0, filter_desc, input1, conv_desc, tensor_desc_1, output0,
			4, &returnedAlgoCount, perfResults, workspace_ptr, workspace_size_in_bytes, false));
		for (int i = 1; i < returnedAlgoCount; ++i)
			if (perfResults[i].time < perfResults[fastest].time)
				fastest = i;
		fprintf(stderr, "[MIOpen] Using algorithim: %s\n", (perfResults[fastest].fwd_algo == miopenConvolutionFwdAlgoDirect ? "Direct" : (perfResults[fastest].fwd_algo == miopenConvolutionFwdAlgoWinograd ? "Winograd" : std::to_string((int)perfResults[fastest].fwd_algo).c_str())));
		workspace_size_in_bytes = perfResults[fastest].memory;
		CUDA_SAFE_CALL(cudaMalloc(&workspace_ptr, workspace_size_in_bytes));
        } catch (...) { fprintf(stderr, "[MIOpen] No any algorithm supports this Conv2D: (@N@, @CI@, @H@, @W@) -> (@N@, @CO@, @HO@, @WO@), pad = (@P0@, @P1@), srd = (@S0@, @S1@).\n"); abort(); }
	}
	CUDNN_SAFE_CALL(miopenConvolutionForward(cudnn_handle,
		&alpha, tensor_desc_0, input0, filter_desc, input1, conv_desc, perfResults[fastest].fwd_algo,
		&beta, tensor_desc_1, output0, workspace_ptr, workspace_size_in_bytes));
	CUDNN_SAFE_CALL(miopenDestroyTensorDescriptor(tensor_desc_0));
	CUDNN_SAFE_CALL(miopenDestroyTensorDescriptor(tensor_desc_1));
	CUDNN_SAFE_CALL(miopenDestroyTensorDescriptor(filter_desc));
	CUDNN_SAFE_CALL(miopenDestroyConvolutionDescriptor(conv_desc));
)",
                                                                 j_map);
                        lu << str;
                    }

                    return _lu;
                }

                LanguageUnit_p emit_dependency() override
                {
                    LanguageUnit_p _lu(new LanguageUnit(get_function_name() + "_dep"));

                    _lu->require(header::cudnn);
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

    } // namespace kernels
} // namespace nnfusion

using namespace nnfusion;
using namespace nnfusion::kernels;

REGISTER_KERNEL_EMITTER(
    "Convolution",                                                                 // op_name
    Device(ROCM_GPU).TypeConstraint(element::f32).Tag("cudnn_kernel").Priority(2), // attrs
    cuda::RocmConvolutionCudnn)                                                    // constructor

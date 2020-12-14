// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "../../cuda_emitter.hpp"
#include "../../cuda_langunit.hpp"
#include "nnfusion/core/operators/generic_op/generic_op.hpp"

DECLARE_bool(frocm_fixed_kernels);

namespace nnfusion
{
    namespace kernels
    {
        namespace cuda
        {
            class ConvFwdFixed : public CudaEmitter
            {
                shared_ptr<nnfusion::op::GenericOp> generic_op;

            public:
                ConvFwdFixed(shared_ptr<KernelContext> ctx)
                    : CudaEmitter(ctx)
                    , generic_op(
                          static_pointer_cast<nnfusion::op::GenericOp>(ctx->gnode->get_op_ptr()))
                {
                    GENERIC_OP_LOGGING();
                }

                LanguageUnit_p emit_function_body() override
                {
                    bool using_fixed = FLAGS_frocm_fixed_kernels;
                    if (!using_fixed)
                        return nullptr;

                    GENERIC_OP_LOGGING();
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

                    // generic_op->validate_and_infer_types();
                    // auto& cfg = generic_op->localOpConfig.getRoot();
                    auto matching =
                        [&](const nnfusion::Shape& _input_shape,
                            const nnfusion::Shape& _filter_shape,
                            const nnfusion::Shape& _output_shape,
                            const nnfusion::Strides& _dilation,
                            const nnfusion::Strides& _data_dilation,
                            const nnfusion::Strides& _stride,
                            const nnfusion::CoordinateDiff& _padding_below_diff,
                            const nnfusion::CoordinateDiff& _padding_above_diff) -> bool {
                        if (input_shape != _input_shape)
                            return false;
                        if (filter_shape != _filter_shape)
                            return false;
                        if (output_shape != _output_shape)
                            return false;
                        if (window_dilation_strides != _dilation)
                            return false;
                        if (data_dilation_strides != _data_dilation)
                            return false;
                        if (window_movement_strides != _stride)
                            return false;
                        if (padding_below_diff != _padding_below_diff)
                            return false;
                        if (padding_above_diff != _padding_above_diff)
                            return false;
                        return true;
                    };
                    std::string templ;
                    if (matching({1, 32, 170, 96},
                                 {32, 32, 21, 11},
                                 {1, 32, 75, 86},
                                 {1, 1},
                                 {1, 1},
                                 {2, 1},
                                 {0, 0},
                                 {0, 0}))
                    {
                        templ =
                            "rocm_adapter/fixed_kernels/convfwd/"
                            "conv2d_fwd_1_32_170_96_32_21-11_2-1_0-0.h.in";
                        m_gridDim = dim3(86, 1, 1);
                        m_blockDim = dim3(1, 15, 32);
                    }
                    else if (matching({128, 3, 227, 227},
                                      {96, 3, 11, 11},
                                      {128, 96, 55, 55},
                                      {1, 1},
                                      {1, 1},
                                      {4, 4},
                                      {0, 0},
                                      {0, 0}))
                    {
                        templ =
                            "rocm_adapter/fixed_kernels/convfwd/"
                            "conv2d_fwd_128_3_227_227_96_11_11_4_0_1.h.in";
                        m_gridDim = dim3(1, 55, 128);
                        m_blockDim = dim3(5, 1, 48);
                    }
                    else
                        return nullptr;

                    LanguageUnit_p _lu(new LanguageUnit(get_function_name()));
                    auto& lu = *_lu;
                    lu.block_begin();
                    lu << nnfusion::codegen::get_content_from_templates(templ) << "\n";
                    lu.block_end();
                    return _lu;
                }

                LanguageUnit_p emit_dependency() override
                {
                    GENERIC_OP_LOGGING();

                    LanguageUnit_p _lu(new LanguageUnit(get_function_name() + "_dep"));
                    _lu->require(header::cuda);
                    return _lu;
                }

                void set_launch_config() override {}
            };
        } // namespace cuda
    }     // namespace kernels
} // namespace nnfusion

using namespace nnfusion;
using namespace nnfusion::kernels;

REGISTER_KERNEL_EMITTER(
    "Convolution",                                                                // op_name
    Device(ROCM_GPU).TypeConstraint(element::f32).Tag("cuda_kernel").Priority(2), // attrs
    cuda::ConvFwdFixed)                                                           // constructor

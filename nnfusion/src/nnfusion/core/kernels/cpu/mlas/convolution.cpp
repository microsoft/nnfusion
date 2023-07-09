// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "convolution.hpp"
#include "nnfusion/common/common.hpp"
#include "nnfusion/core/operators/generic_op/generic_op.hpp"

using namespace nnfusion;
using namespace nnfusion::kernels;

cpu::ConvolutionMlas::ConvolutionMlas(shared_ptr<KernelContext> ctx)
    : MlasKernelEmitter(ctx)
{
    auto conv = static_pointer_cast<op::Convolution>(ctx->gnode->get_op_ptr());

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
    tag << "mlas_convolution_op_" << dtype << "_i" << join(input_shape, "_") << "_w"
        << join(filter_shape, "_") << "_o" << join(output_shape, "_") << "_ws"
        << join(window_movement_strides, "_") << "_wd" << join(window_dilation_strides, "_")
        << "_pb" << join(padding_below_diff, "_") << "_pa" << join(padding_above_diff, "_");
    custom_tag = tag.str();
}

LanguageUnit_p cpu::ConvolutionMlas::emit_function_body()
{
    if (!(data_format == "NCW" || data_format == "NCHW"))
    {
        return nullptr;
    }

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

    size_t batch_count = input_shape[0];
    size_t input_channels = input_shape[1];
    size_t input_height = input_shape[2];
    size_t input_width = input_shape[3];
    size_t filter_count = filter_shape[0];
    size_t kernel_height = filter_shape[2];
    size_t kernel_width = filter_shape[3];
    size_t padding_left_height = padding_below_diff[1];
    size_t padding_left_width = padding_below_diff[0];
    size_t padding_right_height = padding_above_diff[1];
    size_t padding_right_width = padding_above_diff[0];
    size_t dilation_height = window_dilation_strides[0];
    size_t dilation_width = window_dilation_strides[1];
    size_t stride_height = window_movement_strides[0];
    size_t stride_width = window_movement_strides[1];
    size_t output_height = output_shape[2];
    size_t output_width = output_shape[3];

    auto code = op::create_code_from_template(
        R"(
int64_t batch_count = @batch_count@;
int64_t group_count = 1;
int64_t input_channels = @input_channels@;
int64_t input_height = @input_height@;
int64_t input_width = @input_width@;
int64_t filter_count = @filter_count@;
int64_t kernel_height = @kernel_height@;
int64_t kernel_width = @kernel_width@;
int64_t padding_left_height = @padding_left_height@;
int64_t padding_left_width = @padding_left_width@;
int64_t padding_right_height = @padding_right_height@;
int64_t padding_right_width = @padding_right_width@;
int64_t dilation_height = @dilation_height@;
int64_t dilation_width = @dilation_width@;
int64_t stride_height = @stride_height@;
int64_t stride_width = @stride_width@;
int64_t output_height = @output_height@;
int64_t output_width = @output_width@;

int64_t input_shape[] = { input_height, input_width };
int64_t kernel_shape[] = { kernel_height, kernel_width };
int64_t dilation_shape[] = { dilation_height, dilation_width };
int64_t padding[] = { padding_left_height, padding_left_width, padding_right_height, padding_right_width };
int64_t stride_shape[] = { stride_height, stride_width };
int64_t output_shape[] = { output_height, output_width };

MLAS_ACTIVATION activation;
activation.ActivationKind = MlasIdentityActivation;

MLAS_CONV_PARAMETERS parameters;
size_t working_buffer_size = 0;

MlasConvPrepare(&parameters,
                2,
                batch_count,
                group_count,
                input_channels,
                input_shape,
                kernel_shape,
                dilation_shape,
                padding,
                stride_shape,
                output_shape,
                filter_count,
                &activation,
                &working_buffer_size,
                thread_pool);

float* working_buffer = new float[working_buffer_size];

MlasConv(&parameters,
         input0,
         input1,
         nullptr,
         working_buffer,
         output0,
         thread_pool);

delete[] working_buffer;

)",
        {{"batch_count", batch_count},
         {"input_channels", input_channels},
         {"input_height", input_height},
         {"input_width", input_width},
         {"filter_count", filter_count},
         {"kernel_height", kernel_height},
         {"kernel_width", kernel_width},
         {"padding_left_height", padding_left_height},
         {"padding_left_width", padding_left_width},
         {"padding_right_height", padding_right_height},
         {"padding_right_width", padding_right_width},
         {"dilation_height", dilation_height},
         {"dilation_width", dilation_width},
         {"stride_height", stride_height},
         {"stride_width", stride_width},
         {"output_height", output_height},
         {"output_width", output_width}});

    lu << code;

    return _lu;
}

LanguageUnit_p cpu::ConvolutionMlas::emit_dependency()
{
    LanguageUnit_p _lu(new LanguageUnit(get_function_name() + "_dep"));
    _lu->require(header::mlas);

    return _lu;
}

REGISTER_KERNEL_EMITTER(
    "Convolution",                                                            // op_name
    Device(GENERIC_CPU).TypeConstraint(element::f32).Tag("mlas").Priority(6), // attrs
    cpu::ConvolutionMlas)                                                     // constructor

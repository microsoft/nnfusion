// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "max_pool.hpp"
#include "nnfusion/core/operators/generic_op/generic_op.hpp"

using namespace nnfusion;
using namespace nnfusion::kernels;

cpu::MaxPoolMlas::MaxPoolMlas(shared_ptr<KernelContext> ctx)
    : MlasKernelEmitter(ctx)
{
    auto max_pool = static_pointer_cast<op::MaxPool>(ctx->gnode->get_op_ptr());

    input_shape = ctx->inputs[0]->get_shape();
    output_shape = ctx->outputs[0]->get_shape();
    window_shape = max_pool->get_window_shape();
    padding_below = max_pool->get_padding_below();
    padding_above = max_pool->get_padding_above();
    window_stride = max_pool->get_window_movement_strides();
    data_format = max_pool->get_data_format();
    dtype = ctx->outputs[0]->get_element_type().c_type_string();

    std::stringstream tag;
    tag << "mlas_maxpool_" << dtype << "_i" << join(input_shape, "_") << "_w"
        << join(window_shape, "_") << "_o" << join(output_shape, "_") << "_ws"
        << join(window_stride, "_") << "_pb" << join(padding_below, "_") << "_pa"
        << join(padding_above, "_");
    custom_tag = tag.str();
}

LanguageUnit_p cpu::MaxPoolMlas::emit_function_body()
{
    if (data_format == "NHWC")
    {
        return nullptr;
    }

    // emit code
    LanguageUnit_p _lu(new LanguageUnit(get_function_name()));
    auto& lu = *_lu;

    size_t batch_count = input_shape[0];
    size_t input_channels = input_shape[1];
    size_t input_height = input_shape[2];
    size_t input_width = input_shape[3];
    size_t kernel_height = window_shape[0];
    size_t kernel_width = window_shape[1];
    size_t padding_left_height = padding_below[1];
    size_t padding_left_width = padding_below[0];
    size_t padding_right_height = padding_above[1];
    size_t padding_right_width = padding_above[0];
    size_t stride_height = window_stride[0];
    size_t stride_width = window_stride[1];
    size_t output_height = output_shape[2];
    size_t output_width = output_shape[3];

    auto code = op::create_code_from_template(
        R"(
int64_t batch_count = @batch_count@;
int64_t input_channels = @input_channels@;
int64_t input_height = @input_height@;
int64_t input_width = @input_width@;
int64_t kernel_height = @kernel_height@;
int64_t kernel_width = @kernel_width@;
int64_t padding_left_height = @padding_left_height@;
int64_t padding_left_width = @padding_left_width@;
int64_t padding_right_height = @padding_right_height@;
int64_t padding_right_width = @padding_right_width@;
int64_t stride_height = @stride_height@;
int64_t stride_width = @stride_width@;
int64_t output_height = @output_height@;
int64_t output_width = @output_width@;

int64_t input_shape[] = { int64_t(batch_count), int64_t(input_channels), int64_t(input_height), int64_t(input_width) };
int64_t kernel_shape[] = { int64_t(kernel_height), int64_t(kernel_width) };
int64_t padding[] = { int64_t(padding_left_height), int64_t(padding_left_width), int64_t(padding_right_height), int64_t(padding_right_width) };
int64_t stride_shape[] = { int64_t(stride_height), int64_t(stride_width) };
int64_t output_shape[] = { int64_t(batch_count), int64_t(input_channels), int64_t(output_height), int64_t(output_width) };

MlasPool(MlasMaximumPooling, 2, input_shape, kernel_shape, padding, stride_shape, output_shape, input0, output0, thread_pool);

)",
        {{"batch_count", batch_count},
         {"input_channels", input_channels},
         {"input_height", input_height},
         {"input_width", input_width},
         {"kernel_height", kernel_height},
         {"kernel_width", kernel_width},
         {"padding_left_height", padding_left_height},
         {"padding_left_width", padding_left_width},
         {"padding_right_height", padding_right_height},
         {"padding_right_width", padding_right_width},
         {"stride_height", stride_height},
         {"stride_width", stride_width},
         {"output_height", output_height},
         {"output_width", output_width}});

    lu << code;

    return _lu;
}

LanguageUnit_p cpu::MaxPoolMlas::emit_dependency()
{
    LanguageUnit_p _lu(new LanguageUnit(get_function_name() + "_dep"));
    _lu->require(header::mlas);

    return _lu;
}

REGISTER_KERNEL_EMITTER(
    "MaxPool",                                                                // op_name
    Device(GENERIC_CPU).TypeConstraint(element::f32).Tag("mlas").Priority(6), // attrs
    cpu::MaxPoolMlas)                                                         // constructor

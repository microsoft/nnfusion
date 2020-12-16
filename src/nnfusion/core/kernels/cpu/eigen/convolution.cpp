// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "../cpu_kernel_emitter.hpp"
#include "nnfusion/core/operators/generic_op/generic_op.hpp"

namespace nnfusion
{
    namespace kernels
    {
        namespace cpu
        {
            class ConvolutionEigen : public EigenKernelEmitter
            {
            public:
                ConvolutionEigen(shared_ptr<KernelContext> ctx)
                    : EigenKernelEmitter(ctx)
                {
                    auto conv = static_pointer_cast<op::Convolution>(ctx->gnode->get_op_ptr());

                    input_shape = ctx->inputs[0]->get_shape();
                    filter_shape = ctx->inputs[1]->get_shape();
                    output_shape = ctx->outputs[0]->get_shape();
                    window_dilation_strides = conv->get_window_dilation_strides();
                    window_movement_strides = conv->get_window_movement_strides();
                    padding_below_diff = conv->get_padding_below();
                    padding_above_diff = conv->get_padding_above();
                    data_format = conv->get_data_format();
                    dtype = ctx->outputs[0]->get_element_type().c_type_string();

                    std::stringstream tag;
                    tag << "eigen_convolution_op_" << dtype << "_i" << join(input_shape, "_")
                        << "_w" << join(filter_shape, "_") << "_o" << join(output_shape, "_")
                        << "_ws" << join(window_movement_strides, "_") << "_wd"
                        << join(window_dilation_strides, "_") << "_pb"
                        << join(padding_below_diff, "_") << "_pa" << join(padding_above_diff, "_")
                        << "_df" << data_format;
                    custom_tag = tag.str();
                }

                LanguageUnit_p emit_function_body() override
                {
                    if (data_format != "NHWC")
                    {
                        return nullptr;
                    }

                    LanguageUnit_p _lu(new LanguageUnit(get_function_name()));
                    auto& lu = *_lu;

                    uint32_t rank = static_cast<uint32_t>(input_shape.size());
                    int64_t padding[] = {padding_below_diff[1],
                                         padding_above_diff[1],
                                         padding_below_diff[0],
                                         padding_above_diff[0]};

                    auto code = nnfusion::op::create_code_from_template(
                        R"(
Eigen::array<Eigen::IndexPair<Eigen::Index>, 1> contract_dims;
contract_dims[0] = Eigen::IndexPair<Eigen::Index>(1, 0);

Eigen::array<Eigen::Index, @Rank@> in_dims({@in_dims@});
Eigen::array<Eigen::Index, @Rank@> out_dims({@out_dims@});
Eigen::array<Eigen::Index, @Rank@> kernel_dims({@kernel_dims@});
Eigen::DSizes<Eigen::Index, 2> pre_contract_dims;
pre_contract_dims[1] = kernel_dims[2] * kernel_dims[1] * kernel_dims[0];
pre_contract_dims[0] = out_dims[1] * out_dims[2];
for (int i = 0; i < @Rank@ - 3; ++i) {
  pre_contract_dims[0] *= in_dims[i];
}

Eigen::DSizes<Eigen::Index, @Rank@> post_contract_dims;
post_contract_dims[@Rank@- 1] = kernel_dims[3];
post_contract_dims[@Rank@ - 2] = out_dims[2];
post_contract_dims[@Rank@ - 3] = out_dims[1];
for (int i = 0; i < @Rank@ - 3; ++i) {
  post_contract_dims[i] = in_dims[i];
}

Eigen::DSizes<Eigen::Index, 2> new_kernel_dims;
new_kernel_dims[0] = kernel_dims[2] * kernel_dims[1] * kernel_dims[0];
new_kernel_dims[1] = kernel_dims[3];

Eigen::TensorMap<Eigen::Tensor<const @ElementType@, @Rank@, Eigen::RowMajor, Eigen::DenseIndex>, Eigen::Aligned>
    in(static_cast<@ElementType@ *>(input0), in_dims),
    kernel(static_cast<@ElementType@ *>(input1), kernel_dims);
Eigen::TensorMap<Eigen::Tensor<@ElementType@, @Rank@, Eigen::RowMajor, Eigen::DenseIndex>, Eigen::Aligned>
    out(static_cast<@ElementType@ *>(output0), out_dims);

out.device(*(thread_pool->GetDevice())) = in
    .extract_image_patches(kernel_dims[1], kernel_dims[0], @row_stride@,
                           @col_stride@, @row_dilation@, @col_dilation@,
                           /*row_inflate_stride=*/1,
                           /*col_inflate_stride=*/1, @padding@,
                           /*padding_value=*/0)
    .reshape(pre_contract_dims)
    .contract(kernel.reshape(new_kernel_dims), contract_dims)
    .reshape(post_contract_dims);

//    .extract_image_patches(kernel_dims[1], kernel_dims[0], @row_stride@,
//                           @col_stride@, @row_dilation@, @col_dilation@,
//                           Eigen::PADDING_VALID)
//    .reshape(pre_contract_dims)
//    .contract(kernel.reshape(new_kernel_dims), contract_dims)
//    .reshape(post_contract_dims);
)",
                        {{"Rank", rank},
                         {"in_dims", join(input_shape)},
                         {"out_dims", join(output_shape)},
                         {"kernel_dims", join(filter_shape)},
                         {"row_stride", window_movement_strides[1]},
                         {"col_stride", window_movement_strides[0]},
                         {"row_dilation", window_dilation_strides[1]},
                         {"col_dilation", window_dilation_strides[0]},
                         {"padding", join(padding)},
                         {"func_name", get_function_name()},
                         {"ElementType", dtype}});
                    lu << code;

                    return _lu;
                }

                LanguageUnit_p emit_dependency() override
                {
                    LanguageUnit_p _lu(new LanguageUnit(get_function_name() + "_dep"));
                    _lu->require(header::eigen_tensor);
                    _lu->require(header::eigen_spatial_convolution);

                    return _lu;
                }

            private:
                nnfusion::Shape input_shape, filter_shape, output_shape;
                nnfusion::Strides window_dilation_strides, window_movement_strides;
                nnfusion::CoordinateDiff padding_below_diff, padding_above_diff;
                string dtype, data_format;
            };
        } // namespace cpu
    }     // namespace kernels
} // namespace nnfusion

using namespace nnfusion;
using namespace nnfusion::kernels;

REGISTER_KERNEL_EMITTER(
    "Convolution",                                                             // op_name
    Device(GENERIC_CPU).TypeConstraint(element::f32).Tag("eigen").Priority(4), // attrs
    cpu::ConvolutionEigen)

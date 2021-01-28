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
            class MaxPoolEigen : public EigenKernelEmitter
            {
            public:
                MaxPoolEigen(shared_ptr<KernelContext> ctx)
                    : EigenKernelEmitter(ctx)
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
                    tag << "eigen_maxpool_" << dtype << "_i" << join(input_shape, "_") << "_w"
                        << join(window_shape, "_") << "_o" << join(output_shape, "_") << "_ws"
                        << join(window_stride, "_") << "_pb" << join(padding_below, "_") << "_pa"
                        << join(padding_above, "_");
                    custom_tag = tag.str();
                }

                LanguageUnit_p emit_function_body() override
                {
                    if (data_format == "NCHW")
                    {
                        return nullptr;
                    }

                    LanguageUnit_p _lu(new LanguageUnit(get_function_name()));
                    auto& lu = *_lu;

                    auto code = nnfusion::op::create_code_from_template(
                        R"(
const int64_t min_cost_per_rank = 10000;
int num_shards =
    std::max(std::min(static_cast<int64_t>(thread_pool->NumThreads()),
                      @input_size@ / min_cost_per_rank),
             static_cast<int64_t>(1));
const int32_t batch = @batch@;
const int64_t block_size = (batch + num_shards - 1) / num_shards;
if (block_size > batch)
{
    num_shards = 1;
}

const int32_t in_rows = @in_rows@;
const int32_t in_cols = @in_cols@;
const int32_t channel = @channel@;
const int32_t pad_rows = @pad_rows@;
const int32_t pad_cols = @pad_cols@;
const int32_t window_rows = @window_rows@;
const int32_t window_cols = @window_cols@;
const int32_t row_stride = @row_stride@;
const int32_t col_stride = @col_stride@;
const int32_t out_height = @out_height@;
const int32_t out_width = @out_width@;

typedef Eigen::Map<const Eigen::Matrix<@ElementType@, Eigen::Dynamic, Eigen::Dynamic>>
    ConstEigenMatrixMap;
typedef Eigen::Map<Eigen::Matrix<@ElementType@, Eigen::Dynamic, Eigen::Dynamic>>
    EigenMatrixMap;

ConstEigenMatrixMap in_mat(input0, channel, in_cols * in_rows * batch);
EigenMatrixMap out_mat(output0, channel, out_width * out_height * batch);

const int64_t output_image_size = out_height * out_width * channel;

auto func = [&](int __rank__)
{
    int64_t start = block_size * __rank__;
    int64_t end = std::min(start + block_size, static_cast<int64_t>(batch));
    if (start < end)
    {
        EigenMatrixMap out_shard(output0 + start * output_image_size, 1, (end - start) * output_image_size);
        out_shard.setConstant(Eigen::NumTraits<@ElementType@>::lowest());
        for (int32_t b = start; b < end; ++b)
        {
            const int32_t out_offset_batch = b * out_height;
            for (int32_t h = 0; h < in_rows; ++h)
            {
                for (int32_t w = 0; w < in_cols; ++w)
                {
                    const int32_t hpad = h + pad_rows;
                    const int32_t wpad = w + pad_cols;
                    const int32_t h_start = (hpad < window_rows)
                                              ? 0
                                              : (hpad - window_rows) / row_stride + 1;
                    const int32_t h_end = std::min(hpad / row_stride + 1, out_height);
                    const int32_t w_start = (wpad < window_cols)
                                              ? 0
                                              : (wpad - window_cols) / col_stride + 1;
                    const int32_t w_end = std::min(wpad / col_stride + 1, out_width);
                    // compute elementwise max.
                    const int32_t in_offset = (b * in_rows + h) * in_cols + w;
                    for (int32_t ph = h_start; ph < h_end; ++ph)
                    {
                        const int32_t out_offset_base =
                            (out_offset_batch + ph) * out_width;
                        for (int32_t pw = w_start; pw < w_end; ++pw)
                        {
                            const int32_t out_offset = out_offset_base + pw;
                            out_mat.col(out_offset) =
                                out_mat.col(out_offset).cwiseMax(in_mat.col(in_offset));
                        }
                    }
                }
            }
        }
    }
};

thread_pool->ParallelFor(num_shards, func);
)",
                        {{"batch", input_shape[0]},
                         {"in_rows", input_shape[1]},
                         {"in_cols", input_shape[2]},
                         {"channel", input_shape[3]},
                         {"input_size", shape_size(input_shape)},
                         {"pad_rows", padding_below[1]},
                         {"pad_cols", padding_below[0]},
                         {"window_rows", window_shape[0]},
                         {"window_cols", window_shape[1]},
                         {"row_stride", window_stride[0]},
                         {"col_stride", window_stride[1]},
                         {"out_height", output_shape[1]},
                         {"out_width", output_shape[2]},
                         {"ElementType", dtype}});
                    lu << code;

                    return _lu;
                }

                LanguageUnit_p emit_dependency() override
                {
                    LanguageUnit_p _lu(new LanguageUnit(get_function_name() + "_dep"));
                    _lu->require(header::eigen_tensor);

                    return _lu;
                }

            private:
                nnfusion::Shape input_shape, output_shape, window_shape;
                nnfusion::Shape padding_below, padding_above;
                nnfusion::Strides window_stride;
                string dtype, data_format;
            };
        } // namespace cpu
    }     // namespace kernels
} // namespace nnfusion

using namespace nnfusion;
using namespace nnfusion::kernels;

REGISTER_KERNEL_EMITTER(
    "MaxPool",                                                                 // op_name
    Device(GENERIC_CPU).TypeConstraint(element::f32).Tag("eigen").Priority(4), // attrs
    cpu::MaxPoolEigen)

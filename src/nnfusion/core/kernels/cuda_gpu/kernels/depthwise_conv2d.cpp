// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "depthwise_conv2d.hpp"
#include "nnfusion/core/operators/generic_op/generic_op.hpp"

using namespace nnfusion;
using namespace nnfusion::kernels;

cuda::DepthwiseConv2dNative::DepthwiseConv2dNative(shared_ptr<KernelContext> ctx)
    : BlockCudaEmitter(ctx)
{
    auto op = static_pointer_cast<nnfusion::op::GenericOp>(ctx->gnode->get_op_ptr());

    const Shape input_shape = Shape(ctx->inputs[0]->get_shape());
    // [ filter_rows, filter_cols, in_depth, depth_multiplier]
    const Shape filter_shape = Shape(ctx->inputs[1]->get_shape());
    const Shape output_shape = Shape(ctx->outputs[0]->get_shape());

    data_format = op->localOpConfig.getRoot()["data_format"];
    std::vector<int32_t> strides = op->localOpConfig.getRoot()["strides"];
    CoordinateDiff padding_before = op->localOpConfig.getRoot()["padding_before"];
    CoordinateDiff padding_after = op->localOpConfig.getRoot()["padding_after"];

    bool is_nhwc = (data_format == "NHWC");

    const int64_t in_depth = is_nhwc ? input_shape[3] : input_shape[1];
    NNFUSION_CHECK(in_depth == filter_shape[2]);
    const int64_t depth_multiplier = filter_shape[3];
    const int64_t out_depth = in_depth * depth_multiplier;
    const int64_t input_rows = is_nhwc ? input_shape[1] : input_shape[2];
    const int64_t input_cols = is_nhwc ? input_shape[2] : input_shape[3];
    const int64_t filter_rows = filter_shape[0];
    const int64_t filter_cols = filter_shape[1];
    const int64_t batch = input_shape[0];

    args.batch = batch;
    args.in_rows = input_rows;
    args.in_cols = input_cols;
    args.in_depth = in_depth;
    args.filter_rows = filter_rows;
    args.filter_cols = filter_cols;
    args.depth_multiplier = depth_multiplier;
    args.stride = strides[0];
    args.pad_rows = padding_before[0];
    args.pad_cols = padding_before[1];
    args.out_rows = is_nhwc ? output_shape[1] : output_shape[2];
    args.out_cols = is_nhwc ? output_shape[2] : output_shape[3];
    args.out_depth = out_depth;
    args.num_outputs = shape_size(output_shape);
}

LanguageUnit_p cuda::DepthwiseConv2dNative::emit_function_body()
{
    if (data_format == "NHWC")
    {
        return emit_DepthwiseConv2dGPUKernelNHWC();
    }
    else if (data_format == "NCHW")
    {
        return emit_DepthwiseConv2dGPUKernelNCHW();
    }
    else
    {
        return nullptr;
    }
};

LanguageUnit_p cuda::DepthwiseConv2dNative::emit_DepthwiseConv2dGPUKernelNHWC()
{
    LanguageUnit_p _lu(new LanguageUnit(get_function_name()));
    auto& lu = *_lu;

    auto src =
        nnfusion::op::create_code_from_template(R"(
    typedef @data_type@ S;
    float *input = input0;
    float *filter = input1;
    float *output = output0;

    const int in_height = @in_rows@;
    const int in_width = @in_cols@;
    const int in_depth = @in_depth@;
    const int filter_height = @filter_rows@;
    const int filter_width = @filter_cols@;
    const int depth_multiplier = @depth_multiplier@;
    const int stride = @stride@;
    const int pad_height = @pad_rows@;
    const int pad_width = @pad_cols@;
    const int out_height = @out_rows@;
    const int out_width = @out_cols@;
    const int out_depth = @out_depth@;
    const int num_outputs = @num_outputs@;

    for (uint32_t thread_id = blockIdx.x * blockDim.x + threadIdx.x; thread_id < num_outputs;
         thread_id += blockDim.x * gridDim.x)
    {
        // Compute the indexes of this thread in the output.
        const int out_channel = (int)thread_id % out_depth;
        const int out_col = (thread_id / out_depth) % out_width;
        const int out_row = (thread_id / out_depth / out_width) % out_height;
        const int batch = thread_id / out_depth / out_width / out_height;
        // Compute the input depth and the index of depth multiplier.
        const int in_channel = out_channel / depth_multiplier;
        const int multiplier = out_channel % depth_multiplier;

        // Decide if all input is valid, if yes, we can skip the boundary checks
        // for each input.
        const int input_row_start = out_row * stride - pad_height;
        const int input_col_start = out_col * stride - pad_width;
        const int input_row_end = input_row_start + filter_height;
        const int input_col_end = input_col_start + filter_width;

        S sum = static_cast<S>(0);

        const int input_offset_temp = in_height * batch;
        if (input_row_start >= 0 && input_col_start >= 0 && input_row_end < in_height &&
            input_col_end < in_width)
        {
            #pragma unroll
            for (int filter_row = 0; filter_row < filter_height; ++filter_row)
            {
                const int in_row = input_row_start + filter_row;
                const int filter_offset_temp = filter_width * filter_row;
                #pragma unroll
                for (int filter_col = 0; filter_col < filter_width; ++filter_col)
                {
                    const int in_col = input_col_start + filter_col;

                    const int input_offset =
                        in_channel + in_depth * (in_col + in_width * (in_row + input_offset_temp));
                    const int filter_offset =
                        multiplier +
                        depth_multiplier *
                            (in_channel + in_depth * (filter_col + filter_offset_temp));
                    sum += static_cast<S>(__ldg(input + input_offset)) *
                           static_cast<S>(__ldg(filter + filter_offset));
                }
            }
        }
        else
        {
            #pragma unroll
            for (int filter_row = 0; filter_row < filter_height; ++filter_row)
            {
                const int in_row = input_row_start + filter_row;
                const int filter_offset_temp = filter_width * filter_row;
                #pragma unroll
                for (int filter_col = 0; filter_col < filter_width; ++filter_col)
                {
                    const int in_col = input_col_start + filter_col;
                    if (in_row >= 0 && in_row < in_height && in_col >= 0 && in_col < in_width)
                    {
                        const int in_col = input_col_start + filter_col;

                        const int input_offset =
                            in_channel +
                            in_depth * (in_col + in_width * (in_row + input_offset_temp));
                        const int filter_offset =
                            multiplier +
                            depth_multiplier *
                                (in_channel + in_depth * (filter_col + filter_offset_temp));
                        sum += static_cast<S>(__ldg(input + input_offset)) *
                               static_cast<S>(__ldg(filter + filter_offset));
                    }
                }
            }
        }
        output[thread_id] = sum;
    }
    )",
                                                {
                                                    {"data_type", m_context->dtypes[0]},
                                                    {"in_rows", args.in_rows},
                                                    {"in_cols", args.in_cols},
                                                    {"in_depth", args.in_depth},
                                                    {"filter_rows", args.filter_rows},
                                                    {"filter_cols", args.filter_cols},
                                                    {"depth_multiplier", args.depth_multiplier},
                                                    {"stride", args.stride},
                                                    {"pad_rows", args.pad_rows},
                                                    {"pad_cols", args.pad_cols},
                                                    {"out_rows", args.out_rows},
                                                    {"out_cols", args.out_cols},
                                                    {"out_depth", args.out_depth},
                                                    {"num_outputs", args.num_outputs},
                                                });
    lu << src;
    return _lu;
}

LanguageUnit_p cuda::DepthwiseConv2dNative::emit_DepthwiseConv2dGPUKernelNCHW()
{
    LanguageUnit_p _lu(new LanguageUnit(get_function_name()));
    auto& lu = *_lu;

    auto src =
        nnfusion::op::create_code_from_template(R"(
    typedef @data_type@ S;
    S *input = input0;
    S *filter = input1;
    S *output = output0;

    const int in_height = @in_rows@;
    const int in_width = @in_cols@;
    const int in_depth = @in_depth@;
    const int filter_height = @filter_rows@;
    const int filter_width = @filter_cols@;
    const int depth_multiplier = @depth_multiplier@;
    const int stride = @stride@;
    const int pad_height = @pad_rows@;
    const int pad_width = @pad_cols@;
    const int out_height = @out_rows@;
    const int out_width = @out_cols@;
    const int out_depth = @out_depth@;
    const int num_outputs = @num_outputs@;

    for (uint32_t thread_id = blockIdx.x * blockDim.x + threadIdx.x; thread_id < num_outputs;
         thread_id += blockDim.x * gridDim.x)
    {
        // Compute the indexes of this thread in the output.
        //
        // We want coalesced reads so we make sure that each warp reads
        // a contiguous chunk of memory.
        //
        // THIS IS PROBABLY WRONG, we are not doing coalesced reads
        // into the input, because of the depth multiplier division...
        const int out_col = (int)thread_id % out_width;
        const int out_row = (thread_id / out_width) % out_height;
        const int out_channel = (thread_id / out_width / out_height) % out_depth;
        const int batch = thread_id / out_width / out_height / out_depth;

        // Compute the input depth and the index of depth multiplier
        // based off the output depth index that this thread is
        // computing n.
        const int in_channel = out_channel / depth_multiplier;
        const int multiplier = out_channel % depth_multiplier;

        // Data is stored in the following format (let's assume we
        // flatten the height and width into one contiguous dimension
        // called "P".
        //
        // B1C1P1 B1C1P2 ..... B1C2P1 B1C2P2 ....
        // B2C1P1 B2C1P2 ..... B2C2P1 B2C2P2 ....
        //
        // Each row contains in_depth * in_height * in_width values
        // for each sample in the batch.
        //
        // We can further flatten it into:
        //
        // B1C1P1 B1C1P2 .....
        // B1C2P1 B1C2P2 ....
        // B2C1P1 B2C1P2 .....
        // B2C2P1 B2C2P2 ....
        //
        // where each row is a contiguous array of all of the spatial
        // pixels for a given batch and input depth.  The following
        // loop #pragma unrolls across the filter dimensions for a given thread,
        // indexing into the filter value and the corresponding input
        // patch.
        //
        // We can compute the index into the patch once right here.
        const int input_offset_temp = (batch * in_depth + in_channel) * (in_height * in_width);

        // Finally, we can iterate over the spatial dimensions and perform the
        // convolution, writing into the output at the end.
        //
        // We perform an additional optimization, where we can determine
        // whether the patch fits within the image indices statically, and
        // avoid boundary checking within the loop.
        const int input_row_start = out_row * stride - pad_height;
        const int input_col_start = out_col * stride - pad_width;
        const int input_row_end = input_row_start + filter_height;
        const int input_col_end = input_col_start + filter_width;

        S sum = static_cast<S>(0);
        if (input_row_start >= 0 && input_col_start >= 0 && input_row_end < in_height &&
            input_col_end < in_width)
        {
            // Loop that doesn't need to check for boundary conditions.
            #pragma unroll
            for (int filter_row = 0; filter_row < filter_height; ++filter_row)
            {
                const int in_row = input_row_start + filter_row;
                const int filter_offset_temp = filter_width * filter_row;
                #pragma unroll
                for (int filter_col = 0; filter_col < filter_width; ++filter_col)
                {
                    const int in_col = input_col_start + filter_col;

                    const int input_offset = (input_offset_temp) + (in_row * in_width) + in_col;
                    const int filter_offset =
                        multiplier +
                        depth_multiplier *
                            (in_channel + in_depth * (filter_col + filter_offset_temp));
                    sum += static_cast<S>(__ldg(input + input_offset)) *
                           static_cast<S>(__ldg(filter + filter_offset));
                }
            }
        }
        else
        {
            // Loop that needs to check for boundary conditions.
            #pragma unroll
            for (int filter_row = 0; filter_row < filter_height; ++filter_row)
            {
                const int in_row = input_row_start + filter_row;
                const int filter_offset_temp = filter_width * filter_row;
                #pragma unroll
                for (int filter_col = 0; filter_col < filter_width; ++filter_col)
                {
                    const int in_col = input_col_start + filter_col;
                    // TODO(vrv): the in_row check can be done outside of this loop;
                    // benchmark both methods to determine the better decision.
                    if (in_row >= 0 && in_row < in_height && in_col >= 0 && in_col < in_width)
                    {
                        const int in_col = input_col_start + filter_col;

                        // input_offset_temp indexes into the start of memory
                        // where the spatial data starts.
                        const int input_offset = (input_offset_temp) + (in_row * in_width) + in_col;

                        const int filter_offset =
                            multiplier +
                            depth_multiplier *
                                (in_channel + in_depth * (filter_col + filter_offset_temp));
                        sum += static_cast<S>(__ldg(input + input_offset)) *
                               static_cast<S>(__ldg(filter + filter_offset));
                    }
                }
            }
        }

        output[thread_id] = static_cast<S>(sum);
    }
    )",
                                                {
                                                    {"data_type", m_context->dtypes[0]},
                                                    {"in_rows", args.in_rows},
                                                    {"in_cols", args.in_cols},
                                                    {"in_depth", args.in_depth},
                                                    {"filter_rows", args.filter_rows},
                                                    {"filter_cols", args.filter_cols},
                                                    {"depth_multiplier", args.depth_multiplier},
                                                    {"stride", args.stride},
                                                    {"pad_rows", args.pad_rows},
                                                    {"pad_cols", args.pad_cols},
                                                    {"out_rows", args.out_rows},
                                                    {"out_cols", args.out_cols},
                                                    {"out_depth", args.out_depth},
                                                    {"num_outputs", args.num_outputs},
                                                });
    lu << src;
    return _lu;
}

void cuda::DepthwiseConv2dNative::set_launch_config()
{
    uint32_t nthreads = args.num_outputs;
    uint32_t block_size_x = 512;
    uint32_t aligned_grid_size_x = align_to_block_size(nthreads, block_size_x);

    m_gridDim = dim3(aligned_grid_size_x, 1, 1);
    m_blockDim = dim3(block_size_x, 1, 1);
}

LanguageUnit_p cuda::DepthwiseConv2dNative::emit_dependency()
{
    LanguageUnit_p _lu(new LanguageUnit(get_function_name() + "_dep"));
    _lu->require(header::cuda);
    return _lu;
}

REGISTER_KERNEL_EMITTER(
    "DepthwiseConv2dNative",                                                      // op_name
    Device(CUDA_GPU).TypeConstraint(element::f32).Tag("cuda_kernel").Priority(2), // attrs
    cuda::DepthwiseConv2dNative)                                                  // constructor

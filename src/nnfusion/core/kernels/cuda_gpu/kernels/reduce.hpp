// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#pragma once
#include "../cuda_emitter.hpp"
#include "../cuda_langunit.hpp"
#include "nnfusion/core/operators/generic_op/generic_op.hpp"

DECLARE_string(fdefault_device);

namespace nnfusion
{
    namespace kernels
    {
        namespace cuda
        {
            template <class T>
            class Reduce : public CudaEmitter
            {
            public:
                Reduce(shared_ptr<KernelContext> ctx)
                    : CudaEmitter(ctx)
                {
                    if (auto reduce =
                            dynamic_pointer_cast<nnfusion::op::Reduce>(ctx->gnode->get_op_ptr()))
                    {
                        reduce_axis = reduce->get_reduction_axes();
                        init_value = " = 0.0";
                    }
                    else if (auto reduce =
                                 dynamic_pointer_cast<nnfusion::op::Max>(ctx->gnode->get_op_ptr()))
                    {
                        reduce_axis = reduce->get_reduction_axes();
                        init_value = " = input0[in_idx]";
                    }
                    else if (auto reduce =
                                 dynamic_pointer_cast<nnfusion::op::Min>(ctx->gnode->get_op_ptr()))
                    {
                        reduce_axis = reduce->get_reduction_axes();
                        init_value = " = input0[in_idx]";
                    }
                    else if (auto reduce = dynamic_pointer_cast<nnfusion::op::Product>(
                                 ctx->gnode->get_op_ptr()))
                    {
                        reduce_axis = reduce->get_reduction_axes();
                        init_value = " = 1.0";
                    }
                    else if (auto reduce =
                                 dynamic_pointer_cast<nnfusion::op::Sum>(ctx->gnode->get_op_ptr()))
                    {
                        reduce_axis = reduce->get_reduction_axes();
                        init_value = " = 0.0";
                    }
                    else if (auto reduce = dynamic_pointer_cast<nnfusion::op::ReduceAny>(
                                 ctx->gnode->get_op_ptr()))
                    {
                        reduce_axis = reduce->get_reduction_axes();
                        init_value = " = false";
                    }
                    else
                    {
                        NNFUSION_CHECK_FAIL() << "incorrect kernel for reduce";
                    }

                    reduce_rank = reduce_axis.size();
                    input_shape = nnfusion::Shape(ctx->inputs[0]->get_shape());
                    output_shape = nnfusion::Shape(ctx->outputs[0]->get_shape());
                    data_bytes = ctx->inputs[0]->get_element_type().size();
                    rank = input_shape.size();
                    out_rank = rank - reduce_rank;
                    reduce_op = CudaOpMap<T>::op;
                    reduction_split_factor = 32;

                    {
                        // calculate reduction_split_number
                        reduction_split_number = 1;
                        uint32_t num_outputs = static_cast<uint32_t>(shape_size(output_shape));
                        if (reduce_op == "add" && num_outputs < 1024 * 128)
                        {
                            // currently, reduction_split only supports add reduce_op in small output_shape
                            uint32_t reduction_loop_size = 1;
                            Shape reduce_flag(rank, 0);
                            for (auto a : reduce_axis)
                            {
                                reduce_flag[a] = 1;
                            }
                            for (int i = 0; i < rank; i++)
                            {
                                if (reduce_flag[i] != 0)
                                {
                                    reduction_loop_size = input_shape[i];
                                }
                            }
                            if ((reduction_loop_size % reduction_split_factor) == 0)
                            {
                                reduction_split_number =
                                    reduction_loop_size / reduction_split_factor;
                            }
                            else
                            {
                                reduction_split_number =
                                    reduction_loop_size / reduction_split_factor + 1;
                            }
                        }
                    }

                    // use to determine if it is RowReduction

                    std::vector<size_t> axes_flag(input_shape.size(), 0);
                    for (auto const& axis : reduce_axis)
                    {
                        axes_flag[axis] = 1;
                    }
                    height = 1;
                    width = 1;
                    is_row_reduction = true;
                    int i = 0;

                    for (; i < axes_flag.size() && (axes_flag[i] == 0 || input_shape[i] == 1); i++)
                    {
                        height *= input_shape[i];
                    }
                    for (; i < axes_flag.size() && (axes_flag[i] == 1 || input_shape[i] == 1); i++)
                    {
                        width *= input_shape[i];
                    }
                    if (i != axes_flag.size())
                        is_row_reduction = false;

                    // `is_row_reduction` is not working on ROCm, using old implementation
                    if (FLAGS_fdefault_device == "ROCm")
                    {
                        is_row_reduction = false;
                    }
                    // current row_reduction implementation is now working for max/min/prod reduction
                    if (reduce_op != "add") // || width % 32 != 0)
                    {
                        is_row_reduction = false;
                    }

                    if (is_row_reduction)
                        expected_block_size =
                            width > 512
                                ? 512
                                : pow(2, static_cast<size_t>(log2(static_cast<float>(width))));

                    uint32_t block_size_x_acc = 256;
                    nthreads_acc = ctx->gpu_num_sm * block_size_x_acc;

                    input_type = ctx->inputs[0]->get_element_type().c_type_string();
                    output_type = ctx->outputs[0]->get_element_type().c_type_string();
                    std::stringstream tag;
                    tag << "cuda"
                        << "_reduce"
                        << "_" << reduce_op << "_" << input_type << "_" << output_type << "_s_"
                        << join(input_shape, "_") << "_axis_" << join(reduce_axis, "_");
                    custom_tag = tag.str();
                }

                LanguageUnit_p emit_function_body() override
                {
                    // Trivial case: no reduction axes.
                    if (reduce_rank == 0 || shape_size(input_shape) == shape_size(output_shape))
                    {
                        return nullptr;
                    }
                    else if (is_row_reduction)
                    {
                        return emit_row_reduction_body();
                    }
                    else if (out_rank != 0)
                    {
                        return emit_function_body_nd();
                    }
                    else
                    {
                        uint32_t nthreads = static_cast<uint32_t>(shape_size(input_shape));
                        // If the data size is large, call reduce_to_scalar_acc first and then reduce_to_scalar.
                        // Otherwise, call reduce to scalar directly.
                        const uint32_t unroll_size = 8;
                        if (nthreads > nthreads_acc * (unroll_size + 1))
                        {
                            // TODO(wenxh): Ignore this Case.
                            return nullptr;
                        }
                        else
                        {
                            return emit_function_body_scalar();
                        }
                    }
                }

                LanguageUnit_p emit_function_body_memcpy()
                {
                    LanguageUnit_p _lu(new LanguageUnit(get_function_name()));
                    auto& lu = *_lu;

                    lu << "if (input0 != output0) {\n"
                       << "   cudaMemcpyAsync(output0, input0, "
                       << static_cast<uint32_t>(shape_size(input_shape)) << " * sizeof("
                       << input_type << ")"
                       << ", cudaMemcpyDeviceToDevice, stream);\n"
                       << "}\n";

                    return _lu;
                }

                LanguageUnit_p emit_row_reduction_body()
                {
                    LanguageUnit_p _lu(new LanguageUnit(get_function_name()));
                    auto& lu = *_lu;
                    auto code = nnfusion::op::create_code_from_template(
                        R"(
int width = @width@;
int block_size = @block_size@;
const int warp_size = @warp_size@;
__shared__ float shm[warp_size];

int thread_idx = threadIdx.x;
int block_idx = blockIdx.x;
int data_idx_offset = block_idx * width;

float val = 0.0;
for (int tidx = thread_idx; tidx < width; tidx += block_size) {
    int data_idx = tidx + data_idx_offset;
    val += input0[data_idx];
}
val = reduceSum(val, thread_idx, block_size, shm);
if (thread_idx == 0) output0[block_idx] = val;
)",
                        {{"width", width}, {"block_size", expected_block_size}, {"warp_size", 32}});

                    lu << code << "\n";
                    return _lu;
                }

                LanguageUnit_p emit_function_body_nd()
                {
                    LanguageUnit_p _lu(new LanguageUnit(get_function_name()));
                    auto& lu = *_lu;

                    Shape reduce_flag(rank, 0);
                    for (auto a : reduce_axis)
                    {
                        reduce_flag[a] = 1;
                    }
                    input_strides = row_major_strides(input_shape);
                    for (int i = 0; i < rank; i++)
                    {
                        if (reduce_flag[i] != 0)
                        {
                            reduce_shape.push_back(input_shape[i]);
                            reduce_strides.push_back(input_strides[i]);
                        }
                        else
                        {
                            non_reduce_strides.push_back(input_strides[i]);
                            //output_shape.push_back(input_shape[i]);
                        }
                    }
                    NVShape output_strides = row_major_strides(output_shape);
                    uint32_t nthreads = static_cast<uint32_t>(shape_size(output_shape));
                    // TODO: currently we set it to 64, will add tuning method later.
                    uint32_t block_size_x = 64;
                    uint32_t aligned_grid_size_x = align_to_block_size(nthreads, block_size_x);

                    // memset for grid reduction atomicAdd
                    if (reduction_split_number > 1)
                    {
                        m_context->outputs[0]->set_memset(true, 0);
                        m_context->outputs[0]->set_persistent(true);
                    }

                    auto expand_vector_uint32 = [](string name, vector<uint32_t>& d) {
                        stringstream ss;
                        for (int i = 0; i < d.size(); i++)
                            ss << "uint32_t " << name << i << " = " << to_string(d[i]) << ";\n";
                        return ss.str();
                    };

                    lu << expand_vector_uint32("out_strides", output_strides);
                    lu << expand_vector_uint32("non_reduce_strides", non_reduce_strides);
                    lu << expand_vector_uint32("reduce_shape", reduce_shape);
                    lu << expand_vector_uint32("reduce_strides", reduce_strides);

                    lu << "uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x; \n";
                    lu << "if (tid < " << nthreads << ")\n";
                    lu.block_begin();
                    {
                        if (out_rank > 0)
                        {
                            lu << "uint32_t dim_idx_generator = tid;\n";
                        }
                        lu << "uint32_t in_idx = 0;\n";

                        // Loop through all reduction axis.
                        for (int64_t i = 0; i < static_cast<int64_t>(out_rank); i++)
                        {
                            lu << "in_idx += (dim_idx_generator / out_strides" << i
                               << ") * non_reduce_strides" << i << ";\n";
                            lu << "dim_idx_generator %= out_strides" << i << ";\n";
                        }

                        lu << output_type << " r" << init_value << ";\n";

                        if (reduction_split_number > 1)
                        {
                            int64_t last_r_idx = static_cast<int64_t>(reduce_rank) - 1;
                            for (int64_t j = 0; j < last_r_idx; j++)
                            {
                                lu << "for(int idx" << j << " = 0; idx" << j << "< reduce_shape"
                                   << j << "; idx" << j << "++)\n";
                                lu.block_begin();
                            }
                            {
                                lu << "uint32_t reduce_idx = in_idx;\n";
                                for (int64_t j = 0; j < last_r_idx; j++)
                                {
                                    lu << "reduce_idx += idx" << j << " * reduce_strides" << j
                                       << ";\n";
                                }
                                lu << "int idx" << last_r_idx << " = 0;\n";
                                lu << "uint32_t step = reduce_strides" << last_r_idx << ";\n";
                                lu << "idx" << last_r_idx << " += " << reduction_split_factor
                                   << " * blockIdx.y;\n";
                                lu << "int idx_end = min(idx" << last_r_idx << " + "
                                   << reduction_split_factor << ", (int)reduce_shape" << last_r_idx
                                   << ");\n";
                                /* // Unroll last reduction axis.
                                uint32_t unroll_num = 8;
                                uint32_t unroll_shift = 3;
                                lu << "for(; idx" << last_r_idx << " < (reduce_shape" << last_r_idx
                                   << " >> " << unroll_shift << "); idx" << last_r_idx << "++)\n";
                                lu.block_begin();
                                {
                                    for (int k = 0; k < unroll_num; k++)
                                    {
                                        lu << "r = " << reduce_op << "(r , input0[reduce_idx]);\n";
                                        lu << "reduce_idx += step;\n";
                                    }
                                }
                                lu.block_end();
                                lu << "idx" << last_r_idx << " <<= " << unroll_shift << ";\n"; */
                                lu << "reduce_idx += idx" << last_r_idx << " * step;\n";
                                lu << "for(; idx" << last_r_idx << " < idx_end; idx" << last_r_idx
                                   << "++)\n";
                                lu.block_begin();
                                {
                                    lu << "r = " << reduce_op << "(r , input0[reduce_idx]);\n";
                                    lu << "reduce_idx += step;\n";
                                }
                                lu.block_end();
                            }
                            for (int64_t j = 0; j < last_r_idx; j++)
                            {
                                lu.block_end();
                            }
                            lu << "atomicAdd(output0 + tid, r);\n";
                        }
                        else
                        {
                            int64_t last_r_idx = static_cast<int64_t>(reduce_rank) - 1;
                            for (int64_t j = 0; j < last_r_idx; j++)
                            {
                                lu << "for(int idx" << j << " = 0; idx" << j << "< reduce_shape"
                                   << j << "; idx" << j << "++)\n";
                                lu.block_begin();
                            }
                            {
                                lu << "uint32_t reduce_idx = in_idx;\n";
                                for (int64_t j = 0; j < last_r_idx; j++)
                                {
                                    lu << "reduce_idx += idx" << j << " * reduce_strides" << j
                                       << ";\n";
                                }
                                lu << "int idx" << last_r_idx << " = 0;\n";
                                lu << "uint32_t step = reduce_strides" << last_r_idx << ";\n";
                                // Unroll last reduction axis.
                                uint32_t unroll_num = 8;
                                uint32_t unroll_shift = 3;
                                lu << "for(; idx" << last_r_idx << " < (reduce_shape" << last_r_idx
                                   << " >> " << unroll_shift << "); idx" << last_r_idx << "++)\n";
                                lu.block_begin();
                                {
                                    for (int k = 0; k < unroll_num; k++)
                                    {
                                        lu << "r = " << reduce_op << "(r , input0[reduce_idx]);\n";
                                        lu << "reduce_idx += step;\n";
                                    }
                                }
                                lu.block_end();
                                lu << "idx" << last_r_idx << " <<= " << unroll_shift << ";\n";
                                lu << "for(; idx" << last_r_idx << " < reduce_shape" << last_r_idx
                                   << "; idx" << last_r_idx << "++)\n";
                                lu.block_begin();
                                {
                                    lu << "r = " << reduce_op << "(r , input0[reduce_idx]);\n";
                                    lu << "reduce_idx += step;\n";
                                }
                                lu.block_end();
                            }
                            for (int64_t j = 0; j < last_r_idx; j++)
                            {
                                lu.block_end();
                            }
                            lu << "output0[tid] = r;\n";
                        }
                    }
                    lu.block_end();

                    return _lu;
                }

                LanguageUnit_p emit_function_body_scalar()
                {
                    LanguageUnit_p _lu(new LanguageUnit(get_function_name()));
                    auto& lu = *_lu;

                    uint32_t nthreads = static_cast<uint32_t>(shape_size(input_shape));
                    uint32_t n = nthreads;
                    uint32_t block_size_x = 1;
                    while (n > 1)
                    {
                        block_size_x <<= 1;
                        n >>= 1;
                    }
                    block_size_x = fmin(512, block_size_x);

                    // TODO (yanhon): if sdata size is not specified, frozen_reduce_sum_2_graph.pb will crash
                    lu << "extern __shared__ " << output_type << " sdata[" << block_size_x
                       << "];\n";
                    //lu << "extern __shared__ " << output_type << " sdata[];\n";
                    lu << "uint32_t tid = threadIdx.x; \n";
                    lu << "uint32_t step = blockDim.x; \n";
                    lu << "sdata[tid] = 0;\n";
                    lu << "uint32_t in_idx = tid;\n";
                    lu << output_type << " r" << init_value << ";\n";
                    lu << "if(in_idx < " << nthreads << ")\n";
                    lu.block_begin();
                    lu << "r = input0[in_idx];\n";
                    lu << "in_idx += step;\n";
                    lu.block_end();
                    // Accumulate reduction to blockDim.x threads.
                    uint32_t unroll_num = 8;
                    lu << "while(in_idx + (step * " << unroll_num - 1 << ") < " << nthreads
                       << ")\n";
                    lu.block_begin();
                    {
                        for (int i = 0; i < unroll_num; i++)
                        {
                            lu << "r = " << reduce_op << "(r , input0[in_idx]);\n";
                            lu << "in_idx += step;\n";
                        }
                    }
                    lu.block_end();
                    lu << "while(in_idx < " << nthreads << ")\n";
                    lu.block_begin();
                    {
                        lu << "r = " << reduce_op << "(r , input0[in_idx]);\n";
                        lu << "in_idx += step;\n";
                    }
                    lu.block_end();

                    // Accumulate 32 threads for each warp.
                    for (int i = 16; i >= 1; i >>= 1)
                    {
                        if (block_size_x > i)
                        {
                            lu << "r = " << reduce_op << "(r, __shfl_down_sync(0xffffffff, r, " << i
                               << ", 32));\n";
                        }
                    }

                    if (block_size_x > 32)
                    {
                        lu << "uint32_t lane_idx = tid & 0x1f; \n";
                        lu << "uint32_t warp_idx = tid >> 5; \n";
                        lu << "if(lane_idx == 0)\n";
                        lu.block_begin();
                        {
                            lu << "sdata[warp_idx] = r;\n";
                        }
                        lu.block_end();
                        lu << "__syncthreads();\n";

                        uint32_t warp_size = block_size_x >> 5;

                        lu << "if(tid < " << warp_size << ")\n";
                        lu.block_begin();
                        {
                            lu << "r = sdata[tid];\n";
                        }
                        lu.block_end();
                        // Accumulate 32 threads.
                        for (int i = 16; i >= 1; i >>= 1)
                        {
                            if (warp_size > i)
                            {
                                lu << "r = " << reduce_op << "(r, __shfl_down_sync(0xffffffff, r, "
                                   << i << ", 32));\n";
                            }
                        }
                    }

                    lu << "if(tid == 0)\n";
                    lu.block_begin();
                    {
                        lu << "output0[0] = r;\n";
                    }
                    lu.block_end();

                    return _lu;
                }

                LanguageUnit_p emit_dependency() override
                {
                    LanguageUnit_p _lu(new LanguageUnit(get_function_name() + "_dep"));

                    _lu->require(header::cuda);
                    _lu->require(header::stdio);
                    _lu->require(macro::MIN);
                    _lu->require(declaration::num_SMs);
                    if (is_row_reduction)
                    {
                        _lu->require(declaration::cuda_reduce_primitive);
                    }

                    if (CudaOpMap<T>::math_kernel != nullptr)
                    {
                        auto math_kernel =
                            get_math_kernel(reduce_op,
                                            CudaOpMap<T>::math_kernel,
                                            vector<string>{input_type, input_type, output_type});
                        NNFUSION_CHECK_NOT_NULLPTR(math_kernel);
                        _lu->require(math_kernel);
                    }

                    return _lu;
                }

                void set_launch_config() override
                {
                    if (is_row_reduction)
                    {
                        m_gridDim = dim3(height, 1, 1);
                        m_blockDim = dim3(expected_block_size, 1, 1);
                    }
                    else if (out_rank != 0)
                    {
                        uint32_t nthreads = static_cast<uint32_t>(shape_size(output_shape));
                        // TODO: currently we set it to 64, will add tuning method later.
                        uint32_t block_size_x = 64;
                        uint32_t aligned_grid_size_x = align_to_block_size(nthreads, block_size_x);

                        m_gridDim = dim3(aligned_grid_size_x, reduction_split_number, 1);
                        m_blockDim = dim3(block_size_x, 1, 1);
                    }
                    else
                    {
                        uint32_t nthreads = static_cast<uint32_t>(shape_size(input_shape));
                        // If the data size is large, call reduce_to_scalar_acc first and then reduce_to_scalar.
                        // Otherwise, call reduce to scalar directly.
                        const uint32_t unroll_size = 8;
                        if (nthreads > nthreads_acc * (unroll_size + 1))
                        {
                            NNFUSION_CHECK_FAIL() << "No support for GPU memory allocation.";
                        }
                        else
                        {
                            uint32_t nthreads = static_cast<uint32_t>(shape_size(input_shape));
                            uint32_t n = nthreads;
                            uint32_t block_size_x = 1;
                            while (n > 1)
                            {
                                block_size_x <<= 1;
                                n >>= 1;
                            }
                            block_size_x = fmin(512, block_size_x);

                            m_gridDim = dim3(1, 1, 1);
                            m_blockDim = dim3(block_size_x, 1, 1);
                        }
                    }
                }

            private:
                shared_ptr<KernelContext> kernel_ctx;
                nnfusion::AxisSet reduce_axis;
                nnfusion::Shape input_shape, output_shape;
                nnfusion::NVShape input_strides, non_reduce_strides, reduce_strides, reduce_shape;
                size_t data_bytes, rank, reduce_rank, out_rank;
                uint32_t nthreads_acc;
                string reduce_op, input_type, output_type, init_value;
                size_t height, width, expected_block_size;
                bool is_row_reduction;
                size_t reduction_split_factor,
                    reduction_split_number; // split reduction axis for column reduction
            };

            template <class T>
            class ReduceMemcpy : public CudaLibEmitter
            {
            public:
                ReduceMemcpy(shared_ptr<KernelContext> ctx)
                    : CudaLibEmitter(ctx)
                {
                    if (auto reduce =
                            dynamic_pointer_cast<nnfusion::op::Reduce>(ctx->gnode->get_op_ptr()))
                    {
                        reduce_axis = reduce->get_reduction_axes();
                    }
                    else if (auto reduce =
                                 dynamic_pointer_cast<nnfusion::op::Max>(ctx->gnode->get_op_ptr()))
                    {
                        reduce_axis = reduce->get_reduction_axes();
                    }
                    else if (auto reduce =
                                 dynamic_pointer_cast<nnfusion::op::Min>(ctx->gnode->get_op_ptr()))
                    {
                        reduce_axis = reduce->get_reduction_axes();
                    }
                    else if (auto reduce = dynamic_pointer_cast<nnfusion::op::Product>(
                                 ctx->gnode->get_op_ptr()))
                    {
                        reduce_axis = reduce->get_reduction_axes();
                    }
                    else if (auto reduce =
                                 dynamic_pointer_cast<nnfusion::op::Sum>(ctx->gnode->get_op_ptr()))
                    {
                        reduce_axis = reduce->get_reduction_axes();
                    }
                    else if (auto reduce = dynamic_pointer_cast<nnfusion::op::ReduceAny>(
                                 ctx->gnode->get_op_ptr()))
                    {
                        reduce_axis = reduce->get_reduction_axes();
                    }
                    else
                    {
                        NNFUSION_CHECK_FAIL() << "incorrect kernel for reduce";
                    }

                    reduce_rank = reduce_axis.size();
                    input_shape = nnfusion::Shape(ctx->inputs[0]->get_shape());
                    output_shape = nnfusion::Shape(ctx->outputs[0]->get_shape());
                    rank = input_shape.size();
                    out_rank = rank - reduce_rank;

                    reduce_op = CudaOpMap<T>::op;
                    input_type = ctx->inputs[0]->get_element_type().c_type_string();
                    output_type = ctx->outputs[0]->get_element_type().c_type_string();

                    std::stringstream tag;
                    tag << "cuda"
                        << "_reduce_Memcpy"
                        << "_" << reduce_op << "_" << input_type << "_" << output_type << "_s_"
                        << join(input_shape, "_") << "_axis_" << join(reduce_axis, "_");
                    custom_tag = tag.str();

                    // add inplace tag
                    if (reduce_rank == 0 || shape_size(input_shape) == shape_size(output_shape))
                    {
                        if (!ctx->annotations)
                        {
                            ctx->annotations = std::make_shared<Annotations>();
                        }
                        ctx->annotations->add_in_place_oi_pair({0, 0, false});
                    }
                }

                bool is_eliminative() override
                {
                    if ((reduce_rank == 0 || shape_size(input_shape) == shape_size(output_shape)) &&
                        m_context->inputs[0]->is_same_address(m_context->outputs[0]))
                        return true;
                    else
                        return false;
                }

                LanguageUnit_p emit_function_body() override
                {
                    if (reduce_rank != 0 && shape_size(input_shape) != shape_size(output_shape))
                    {
                        return nullptr;
                    }

                    LanguageUnit_p _lu(new LanguageUnit(get_function_name()));
                    auto& lu = *_lu;
                    auto& dst = m_context->outputs[0];
                    auto& src = m_context->inputs[0];
                    lu << dst->get_element_type().c_type_string() << "* " << dst->get_name()
                       << " = output0;\n";
                    lu << src->get_element_type().c_type_string() << "* " << src->get_name()
                       << " = input0;\n";

                    //emit_memcpyDtD(lu, dst, src);
                    lu << "if (input0 != output0) {\n"
                       << "    CUDA_SAFE_CALL(cudaMemcpyAsync(" << dst->get_name() << ", "
                       << src->get_name() << ", " << dst->size()
                       << ", cudaMemcpyDeviceToDevice, stream));\n"
                       << "}\n";

                    return _lu;
                }

                LanguageUnit_p emit_dependency() override
                {
                    LanguageUnit_p _lu(new LanguageUnit(get_function_name() + "_dep"));
                    _lu->require(header::cuda);
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
                       << "(cudaStream_t stream, " << join(params, ", ") << ")";
                    return _lu;
                }

            private:
                shared_ptr<KernelContext> kernel_ctx;
                nnfusion::AxisSet reduce_axis;
                nnfusion::Shape input_shape, output_shape;
                size_t rank, reduce_rank, out_rank;
                string reduce_op, input_type, output_type;
            };
        } // namespace cuda
    }     // namespace kernels
} // namespace nnfusion

// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "../cuda_emitter.hpp"
#include "../cuda_langunit.hpp"

namespace nnfusion
{
    namespace kernels
    {
        namespace cuda
        {
            //todo(wenxh): this is not same with our design, need to replace the kernel with CudaEmitter
            class Concat : public KernelEmitter
            {
            public:
                Concat(shared_ptr<KernelContext> ctx)
                    : KernelEmitter(ctx, "cuda_sp")
                {
                    op = static_pointer_cast<nnfusion::op::Concat>(ctx->gnode->get_op_ptr());
                    NNFUSION_CHECK_NOT_NULLPTR(op) << "Node type is not Concat.";

                    this->axis = op->get_concatenation_axis();

                    is_memcpy = true;
                    for (size_t idx = 0; idx < ctx->inputs.size(); idx++)
                    {
                        auto& input_shape = ctx->inputs[idx]->get_shape();
                        for (size_t i = 0; i < axis; i++)
                        {
                            if (input_shape[i] != 1)
                            {
                                is_memcpy = false;
                                break;
                            }
                        }
                        if (!is_memcpy)
                            break;
                    }

                    if (is_memcpy)
                    {
                        size_t offset = 0;
                        size_t data_type_size = ctx->outputs[0]->get_element_type().size();
                        for (size_t idx = 0; idx < ctx->inputs.size(); idx++)
                        {
                            if (!ctx->annotations)
                                ctx->annotations = std::make_shared<Annotations>();
                            ctx->annotations->add_in_place_oi_pair(oi_pair(0, idx, false, offset));
                            auto& input_shape = ctx->inputs[idx]->get_shape();
                            offset += shape_size(input_shape) * data_type_size;
                        }
                    }

                    input_num = ctx->inputs.size();
                    split_input_size =
                        256; //max num of inputs fit 4KB parameter space: 256 * 8 + 7 * ?
                    residue = input_num % split_input_size;

                    inputs_strides = std::vector<uint32_t>(input_num, 1);
                    output_stride = 0;
                    concat_axis = this->axis;

                    for (size_t i = 0; i < input_num; i++)
                    {
                        auto arg_rank = ctx->inputs[i]->get_shape().size();
                        for (size_t j = concat_axis; j < arg_rank; j++)
                        {
                            inputs_strides[i] *= ctx->inputs[i]->get_shape()[j];
                        }
                        output_stride += inputs_strides[i];
                    }

                    block_size_x = 64;
                    split_input_stride_offsets.push_back(0);
                    split_input_stride_offset = 0;

                    for (uint32_t i = 0; i < input_num; i += split_input_size)
                    {
                        uint32_t nthread = 0;
                        uint32_t split_output_stride = 0;
                        for (uint32_t j = i; j < i + split_input_size && j < input_num; j++)
                        {
                            nthread += shape_size(ctx->inputs[j]->get_shape());
                            split_output_stride += inputs_strides[j];
                        }
                        split_input_stride_offset += split_output_stride;
                        split_input_stride_offsets.push_back(split_input_stride_offset);
                        split_output_strides.push_back(split_output_stride);
                        split_nthreads.push_back(static_cast<uint32_t>(nthread));
                        split_aligned_grid_size_x.push_back(
                            align_to_block_size(split_nthreads.back(), block_size_x));
                    }

                    dtype = ctx->outputs[0]->get_element_type().c_type_string();

                    std::stringstream tag;
                    tag << "_s" << join(ctx->outputs[0]->get_shape(), "_") << "_a_" << concat_axis;
                    for (size_t i = 0; i < input_num; i++)
                    {
                        tag << "_i_" << join(ctx->inputs[i]->get_shape(), "_");
                    }
                    custom_tag = tag.str();
                }

                bool is_eliminative() override
                {
                    if (is_memcpy && m_context->inputs[0]->is_same_address(m_context->outputs[0]))
                        return true;
                    else
                        return false;
                }

                LanguageUnit_p emit_function_body() override
                {
                    if (input_num <= 256)
                        return nullptr;
                    LanguageUnit_p _lu(new LanguageUnit(get_function_name()));
                    auto& writer = *_lu;

                    if (is_memcpy)
                    {
                        writer << "if (input0 == output0) {\n";
                        size_t offset = 0;
                        for (int i = 0; i < m_context->inputs.size(); i++)
                        {
                            writer << "  assert(input" << i << " == output0 + " << offset << ");\n";
                            offset += shape_size(m_context->inputs[i]->get_shape());
                        }
                        writer << "  return;\n}\n";
                    }

                    for (uint32_t i = 0, n = 0; i < input_num; i += split_input_size, n++)
                    {
                        std::vector<string> args_list;
                        for (uint32_t j = i; j < i + split_input_size && j < input_num; j++)
                        {
                            args_list.push_back("input" + to_string(j));
                        }
                        args_list.push_back("output0");
                        // args_list.push_back(&param_inputs_strides);
                        args_list.push_back(to_string(output_stride));
                        args_list.push_back(to_string(split_output_strides[n]));
                        args_list.push_back(to_string(split_input_stride_offsets[n]));
                        args_list.push_back(to_string(i));
                        args_list.push_back(to_string(split_nthreads[n]));
                        auto kernel =
                            (args_list.size() == split_input_size + 6) ? "_kernel_0" : "_kernel_1";

                        writer << get_function_name() << kernel << "<<<dim3("
                               << split_aligned_grid_size_x[n] << ", " << 1 << ", " << 1
                               << "), dim3(" << block_size_x << ", " << 1 << ", " << 1 << "), " << 0
                               << ", stream"
                               << ">>>(" << join(args_list) << ");\n";
                    }
                    return _lu;
                }

                LanguageUnit_p emit_dependency() override

                {
                    LanguageUnit_p _lu(new LanguageUnit(get_function_name() + "_dep"));
                    _lu->require(header::cuda);

                    LanguageUnit_p concat_kernels(new LanguageUnit(
                        "declaration::" + get_function_name() + "_concat_private_kernels"));

                    auto& writer = *concat_kernels;

                    if (input_num >= split_input_size)
                    {
                        size_t num_inputs = split_input_size;
                        writer << "extern \"C\" __global__ void " << get_function_name()
                               << "_kernel_0(";
                        for (size_t i = 0; i < num_inputs; i++)
                        {
                            writer << dtype << "* in" << i << ", ";
                        }
                        writer
                            << dtype
                            << "* out, uint32_t output_stride, uint32_t "
                               "split_output_stride, uint32_t split_input_stride_offset, uint32_t "
                               "input_offset, uint32_t n)\n";
                        writer.block_begin();
                        {
                            writer << "uint32_t inputs_strides[] = {" << join(inputs_strides)
                                   << "};\n";
                            writer << "uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;\n";
                            writer << "if(tid < n)\n";
                            writer.block_begin();
                            {
                                writer << "uint32_t block_id = tid / split_output_stride;\n";
                                writer << "uint32_t block_idx = tid % split_output_stride;\n";
                                writer << "uint32_t output_idx = block_id * output_stride + "
                                          "block_idx + "
                                          "split_input_stride_offset;\n";
                                writer << "out[output_idx] = 1;\n";
                                for (size_t i = 0; i < num_inputs; i++)
                                {
                                    writer << "if(block_idx < inputs_strides[" << i
                                           << " + input_offset])\n";
                                    writer.block_begin();
                                    {
                                        writer << "out[output_idx] = in" << i
                                               << "[block_id * inputs_strides[" << i
                                               << " + input_offset] + block_idx];\n";
                                        writer << "return;\n";
                                    }
                                    writer.block_end();
                                    writer << "block_idx -= inputs_strides[" << i
                                           << " + input_offset];\n";
                                }
                            }
                            writer.block_end();
                        }
                        writer.block_end();
                    }

                    if (residue != 0)
                    {
                        size_t num_inputs = residue;
                        writer << "extern \"C\" __global__ void " << get_function_name()
                               << "_kernel_1(";
                        for (size_t i = 0; i < num_inputs; i++)
                        {
                            writer << dtype << "* in" << i << ", ";
                        }
                        writer
                            << dtype
                            << "* out, uint32_t output_stride, uint32_t "
                               "split_output_stride, uint32_t split_input_stride_offset, uint32_t "
                               "input_offset, uint32_t n)\n";
                        writer.block_begin();
                        {
                            writer << "uint32_t inputs_strides[] = {" << join(inputs_strides)
                                   << "};\n";
                            writer << "uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;\n";
                            writer << "if(tid < n)\n";
                            writer.block_begin();
                            {
                                writer << "uint32_t block_id = tid / split_output_stride;\n";
                                writer << "uint32_t block_idx = tid % split_output_stride;\n";
                                writer << "uint32_t output_idx = block_id * output_stride + "
                                          "block_idx + "
                                          "split_input_stride_offset;\n";
                                writer << "out[output_idx] = 1;\n";
                                for (size_t i = 0; i < num_inputs; i++)
                                {
                                    writer << "if(block_idx < inputs_strides[" << i
                                           << " + input_offset])\n";
                                    writer.block_begin();
                                    {
                                        writer << "out[output_idx] = in" << i
                                               << "[block_id * inputs_strides[" << i
                                               << " + input_offset] + block_idx];\n";
                                        writer << "return;\n";
                                    }
                                    writer.block_end();
                                    writer << "block_idx -= inputs_strides[" << i
                                           << " + input_offset];\n";
                                }
                            }
                            writer.block_end();
                        }
                        writer.block_end();
                    }
                    _lu->require(concat_kernels);

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
                shared_ptr<nnfusion::op::Concat> op;

                size_t input_num, split_input_size, residue, concat_axis, split_input_stride_offset;
                std::vector<uint32_t> inputs_strides, split_nthreads, split_output_strides,
                    split_input_stride_offsets, split_aligned_grid_size_x;
                uint32_t output_stride;
                uint32_t block_size_x;
                string dtype;
                size_t axis;
                bool is_memcpy = false;
            };
        } // namespace cuda
    }     // namespace kernels
} // namespace nnfusion

using namespace nnfusion;
using namespace nnfusion::kernels;
REGISTER_KERNEL_EMITTER("Concat",                                      //op_name
                        Device(CUDA_GPU).TypeConstraint(element::f32), //attrs
                        cuda::Concat)                                  // constructor

namespace nnfusion
{
    namespace kernels
    {
        namespace cuda
        {
            class ConcatKernel : public BlockCudaEmitter
            {
            public:
                ConcatKernel(shared_ptr<KernelContext> ctx)
                    : BlockCudaEmitter(ctx)
                {
                    op = static_pointer_cast<nnfusion::op::Concat>(ctx->gnode->get_op_ptr());
                    NNFUSION_CHECK_NOT_NULLPTR(op) << "Node type is not Concat.";

                    this->axis = op->get_concatenation_axis();

                    is_memcpy = true;
                    for (size_t idx = 0; idx < ctx->inputs.size(); idx++)
                    {
                        auto& input_shape = ctx->inputs[idx]->get_shape();
                        for (size_t i = 0; i < axis; i++)
                        {
                            if (input_shape[i] != 1)
                            {
                                is_memcpy = false;
                                break;
                            }
                        }
                        if (!is_memcpy)
                            break;
                    }

                    if (is_memcpy)
                    {
                        size_t offset = 0;
                        size_t data_type_size = ctx->outputs[0]->get_element_type().size();
                        for (size_t idx = 0; idx < ctx->inputs.size(); idx++)
                        {
                            if (!ctx->annotations)
                                ctx->annotations = std::make_shared<Annotations>();
                            ctx->annotations->add_in_place_oi_pair(oi_pair(0, idx, false, offset));
                            auto& input_shape = ctx->inputs[idx]->get_shape();
                            offset += shape_size(input_shape) * data_type_size;
                        }
                    }

                    input_num = ctx->inputs.size();
                    inputs_strides = std::vector<uint32_t>(input_num, 1);
                    output_stride = 0;
                    concat_axis = this->axis;
                    nthreads = 0;

                    for (size_t i = 0; i < input_num; i++)
                    {
                        auto arg_rank = ctx->inputs[i]->get_shape().size();
                        nthreads += shape_size(ctx->inputs[i]->get_shape());
                        for (size_t j = concat_axis; j < arg_rank; j++)
                        {
                            inputs_strides[i] *= ctx->inputs[i]->get_shape()[j];
                        }
                        output_stride += inputs_strides[i];
                    }

                    std::stringstream tag;
                    tag << "_s" << join(ctx->outputs[0]->get_shape(), "_") << "_a_" << concat_axis;
                    for (size_t i = 0; i < input_num; i++)
                    {
                        tag << "_i_" << join(ctx->inputs[i]->get_shape(), "_");
                    }
                    custom_tag = tag.str();
                }

                bool is_eliminative() override
                {
                    if (is_memcpy &&
                        m_context->inputs[0]->get_pool_offset() ==
                            m_context->outputs[0]->get_pool_offset())
                        return true;
                    else
                        return false;
                }

                LanguageUnit_p emit_function_body() override
                {
                    // max num of inputs fit 4KB parameter space
                    if (input_num > 256)
                    {
                        return nullptr;
                    }

                    LanguageUnit_p _lu(new LanguageUnit(get_function_name()));
                    auto& writer = *_lu;

                    writer << "uint32_t inputs_strides[] = {" << join(inputs_strides) << "};\n";
                    writer << "uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;\n";
                    writer << "if(tid < " << nthreads << ")\n";
                    writer.block_begin();
                    {
                        writer << "uint32_t block_id = tid / " << output_stride << ";\n";
                        writer << "uint32_t block_idx = tid % " << output_stride << ";\n";
                        writer << "uint32_t output_idx = block_id * " << output_stride
                               << " + block_idx;\n";
                        //writer << "out[output_idx] = 1;\n";
                        for (size_t i = 0; i < input_num; i++)
                        {
                            writer << "if(block_idx < inputs_strides[" << i << "])\n";
                            writer.block_begin();
                            {
                                writer << "output0[output_idx] = input" << i
                                       << "[block_id * inputs_strides[" << i << "] + block_idx];\n";
                                writer << "return;\n";
                            }
                            writer.block_end();
                            writer << "block_idx -= inputs_strides[" << i << "];\n";
                        }
                    }
                    writer.block_end();
                    return _lu;
                }

                void set_launch_config() override
                {
                    uint32_t block_size_x = 512;
                    uint32_t aligned_grid_size_x = align_to_block_size(nthreads, block_size_x);
                    m_gridDim = dim3(aligned_grid_size_x, 1, 1);
                    m_blockDim = dim3(block_size_x, 1, 1);
                }

                LanguageUnit_p emit_dependency() override
                {
                    LanguageUnit_p _lu(new LanguageUnit(get_function_name() + "_dep"));
                    _lu->require(header::cuda);
                    return _lu;
                }

            private:
                shared_ptr<KernelContext> kernel_ctx;
                shared_ptr<nnfusion::op::Concat> op;

                size_t input_num, concat_axis;
                std::vector<uint32_t> inputs_strides;
                uint32_t nthreads;
                uint32_t output_stride;
                size_t axis;
                bool is_memcpy = false;
            };
        } // namespace cuda
    }     // namespace kernels
} // namespace nnfusion

REGISTER_KERNEL_EMITTER("Concat",                                      //op_name
                        Device(CUDA_GPU).TypeConstraint(element::f32), //attrs
                        cuda::ConcatKernel)                            // constructor

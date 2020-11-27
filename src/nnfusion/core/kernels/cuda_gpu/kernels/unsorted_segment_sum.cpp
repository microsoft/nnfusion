// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

// This is the 2rd-generation of kernel definition, recommend to extend new ops with this style
// Changes needed for creating an new kernel with 2rd generation style.
//

#include "../cuda_emitter.hpp"
#include "../cuda_langunit.hpp"
#include "nnfusion/core/operators/generic_op/generic_op.hpp"

namespace nnfusion
{
    namespace kernels
    {
        namespace cuda
        {
            class UnsortedSegmentSum : public CudaLibEmitter
            {
                shared_ptr<nnfusion::op::GenericOp> generic_op;
                size_t input_size, output_size;
                string input0_t, input1_t, output0_t;
                size_t input_outer_dim_size;
                size_t input_inner_dim_size;
                LanguageUnit_p reset_memory_kernel;
                LanguageUnit_p unsorted_segment_sum_kernel;
                string reset_memory_func_name;
                string unsorted_segment_func_name;

            public:
                UnsortedSegmentSum(shared_ptr<KernelContext> ctx)
                    : CudaLibEmitter(ctx)
                    , generic_op(
                          static_pointer_cast<nnfusion::op::GenericOp>(ctx->gnode->get_op_ptr()))
                {
                    input_size = ctx->inputs[0]->size(false);
                    output_size = ctx->outputs[0]->size(false);
                    input_outer_dim_size = (ctx->inputs[1]->get_shape())[0];
                    input_inner_dim_size = ctx->inputs[0]->size(false) / input_outer_dim_size;

                    input0_t = ctx->inputs[0]->get_element_type().c_type_string();
                    input1_t = ctx->inputs[1]->get_element_type().c_type_string();
                    output0_t = ctx->outputs[0]->get_element_type().c_type_string();

                    std::stringstream ss;
                    ss << "ResetMemory_" << input0_t;
                    reset_memory_func_name = ss.str();

                    ss.str(std::string());
                    ss << "UnsortedSegmentSum_" << input0_t << "_" << input1_t << "_" << output0_t;
                    unsorted_segment_func_name = ss.str();
                }

                void define_kernels()
                {
                    // Define the kernel for memory reset.
                    reset_memory_kernel.reset(
                        new LanguageUnit("declaration::reset_memory_private_kernels"));
                    auto& lu_reset_memory_kernel = *reset_memory_kernel;

                    auto code = nnfusion::op::create_code_from_template(
                        R"(
__global__ void @func_name@(@input0_t@* input0)
{
uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
if(tid >= @nthreads@) return;
input0[tid] = 0;
}
)",
                        {{"func_name", reset_memory_func_name},
                         {"input0_t", input0_t},
                         {"nthreads", output_size}});

                    lu_reset_memory_kernel << code;

                    // Define the kernel for unsorted_segment_sum.
                    unsorted_segment_sum_kernel.reset(
                        new LanguageUnit("declaration::unsorted_segment_sum_private_kernels"));
                    auto& lu_unsorted_seg_sum_kernel = *unsorted_segment_sum_kernel;

                    code = nnfusion::op::create_code_from_template(
                        R"(
__global__ void @func_name@(@input0_t@* input0, @input1_t@* input1, @output0_t@* output0)
{
uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
if(tid >= @nthreads@) return;
size_t input_segment_index = tid / @input_inner_dim_size@;
size_t segment_offset = tid % @input_inner_dim_size@;
size_t output_segment_index = input1[input_segment_index];
size_t output_index = output_segment_index * @input_inner_dim_size@ + segment_offset;
atomicAdd(output0 + output_index, input0[tid]);
}
)",
                        {{"func_name", unsorted_segment_func_name},
                         {"input0_t", input0_t},
                         {"input1_t", input1_t},
                         {"output0_t", output0_t},
                         {"nthreads", input_size},
                         {"input_inner_dim_size", input_inner_dim_size}});

                    lu_unsorted_seg_sum_kernel << code;
                }

                LanguageUnit_p emit_function_body() override
                {
                    define_kernels();

                    LanguageUnit_p _lu(new LanguageUnit(get_function_name()));
                    auto& lu = *_lu;

                    uint32_t block_size_x = 512;

                    size_t output_block_cnt = align_to_block_size(output_size, block_size_x);
                    auto code = nnfusion::op::create_code_from_template(
                        R"(
@func_name@<<<dim3(@block_cnt@, 1, 1), dim3(@block_size_x@, 1, 1), 0, stream>>>(output0);
)",
                        {{"func_name", reset_memory_func_name},
                         {"block_cnt", output_block_cnt},
                         {"block_size_x", block_size_x}});
                    lu << code;

                    size_t input_block_cnt = align_to_block_size(input_size, block_size_x);
                    code = nnfusion::op::create_code_from_template(
                        R"(
@func_name@<<<dim3(@block_cnt@, 1, 1), dim3(@block_size_x@, 1, 1), 0, stream>>>(input0, input1, output0);
)",
                        {{"func_name", unsorted_segment_func_name},
                         {"block_cnt", input_block_cnt},
                         {"block_size_x", block_size_x}});
                    lu << code;
                    return _lu;
                }

                LanguageUnit_p emit_dependency() override
                {
                    LanguageUnit_p _lu(new LanguageUnit(get_function_name() + "_dep"));
                    _lu->require(header::cuda);
                    _lu->require(header::stdio);
                    _lu->require(reset_memory_kernel);
                    _lu->require(unsorted_segment_sum_kernel);
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
            };
        }
    }
}

using namespace nnfusion;
using namespace nnfusion::kernels;

REGISTER_KERNEL_EMITTER(
    "UnsortedSegmentSum",
    Device(CUDA_GPU).TypeConstraint(element::f32).Tag("cuda_kernel").Priority(2),
    cuda::UnsortedSegmentSum)

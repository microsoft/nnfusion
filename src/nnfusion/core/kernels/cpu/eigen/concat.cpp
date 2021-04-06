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
            class ConcatEigen : public EigenKernelEmitter
            {
            public:
                ConcatEigen(shared_ptr<KernelContext> ctx)
                    : EigenKernelEmitter(ctx)
                {
                    auto op = static_pointer_cast<nnfusion::op::Concat>(ctx->gnode->get_op_ptr());
                    NNFUSION_CHECK_NOT_NULLPTR(op) << "Node type is not Concat.";

                    this->concat_axis = op->get_concatenation_axis();

                    is_memcpy = true;
                    for (size_t idx = 0; idx < ctx->inputs.size(); ++idx)
                    {
                        auto& input_shape = ctx->inputs[idx]->get_shape();
                        for (size_t i = 0; i < concat_axis; ++i)
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
                    input_strides = std::vector<uint32_t>(input_num, 1);
                    output_stride = 0;
                    output_size = shape_size(ctx->outputs[0]->get_shape());
                    dtype = ctx->outputs[0]->get_element_type().c_type_string();

                    for (size_t i = 0; i < input_num; i++)
                    {
                        auto arg_rank = ctx->inputs[i]->get_shape().size();
                        for (size_t j = concat_axis; j < arg_rank; ++j)
                        {
                            input_strides[i] *= ctx->inputs[i]->get_shape()[j];
                        }
                        output_stride += input_strides[i];
                    }
                    dim0 = output_size / output_stride;

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
                    LanguageUnit_p _lu(new LanguageUnit(get_function_name()));
                    auto& lu = *_lu;

                    std::vector<std::string> inputs;
                    inputs.reserve(input_num);
                    for (int i = 0; i < input_num; ++i)
                    {
                        inputs.push_back("input" + std::to_string(i));
                    }

                    auto code = nnfusion::op::create_code_from_template(
                        R"(
const int64_t min_cost_per_rank = 4096;
int num_shards =
    std::max(std::min(static_cast<int64_t>(thread_pool->NumThreads()),
                      @output_size@ / min_cost_per_rank),
             static_cast<int64_t>(1));
const int64_t block_size = (@output_size@ + num_shards - 1) / num_shards;
if (block_size > @output_size@)
{
    num_shards = 1;
}
int64_t input_strides[] = {@input_strides@};
int64_t output_stride = @output_stride@;
size_t type_size = sizeof(@ElementType@);

auto func = [&](int __rank__)
{
    int64_t start = block_size * __rank__;
    int64_t end = std::min(start + block_size, static_cast<int64_t>(@output_size@));
    int64_t skipped_rows = start / output_stride;
    @ElementType@* out = output0 + skipped_rows * output_stride;
    @ElementType@* out_start = output0 + start;
    @ElementType@* out_end = output0 + end;
    @ElementType@* inputs[@input_num@] = {@inputs@};

    if (out < out_start)
    {
        for (int i = 0; i < @input_num@; ++i)
        {
            int64_t size = input_strides[i];
            int64_t offset = out_start - out;
            if (size <= offset)
            {
                out += size;
                continue;
            }
            const float* inp = reinterpret_cast<float*>(inputs[i] + (skipped_rows * size));
            if (offset > 0)
            {
                out += offset;
                inp += offset;
                size -= offset;
            }
            size = std::min(size, out_end - out);
            if (size <= 0) break;
            memcpy(out, inp, size);
            out += size;
        }
        ++skipped_rows;
    }
    if (out == out_end) return;

    std::vector<const float*> inp;
    inp.reserve(@input_num@);
    for (int i = 0; i < @input_num@; ++i)
    {
        inp.push_back(reinterpret_cast<float*>(inputs[i] + skipped_rows * input_strides[i]));
    }
    const int64_t dim0 = @dim0@;
    for (int64_t i = skipped_rows; i < dim0; ++i)
    {
        for (int64_t j = 0; j < @input_num@; ++j)
        {
            int64_t size = std::min(input_strides[j], out_end - out);
            memcpy(out, inp[j], size * type_size);
            out += size;
            inp[j] += size;
            if (out == out_end) return;
        }
    }
};

thread_pool->ParallelFor(num_shards, func);
)",
                        {{"output_size", output_size},
                         {"input_strides", join(input_strides)},
                         {"output_stride", output_stride},
                         {"input_num", input_num},
                         {"inputs", join(inputs)},
                         {"dim0", dim0},
                         {"func_name", get_function_name()},
                         {"ElementType", dtype}});
                    lu << code;

                    return _lu;
                }

                LanguageUnit_p emit_dependency() override
                {
                    LanguageUnit_p _lu(new LanguageUnit(get_function_name() + "_dep"));

                    return _lu;
                }

            private:
                size_t concat_axis, input_num;
                std::vector<uint32_t> input_strides;
                uint64_t output_size, output_stride, dim0;
                string dtype;
                bool is_memcpy = false;
            };
        } // namespace cpu
    }     // namespace kernels
} // namespace nnfusion

using namespace nnfusion;
using namespace nnfusion::kernels;

REGISTER_KERNEL_EMITTER(
    "Concat",                                                                  // op_name
    Device(GENERIC_CPU).TypeConstraint(element::f32).Tag("eigen").Priority(4), // attrs
    cpu::ConcatEigen)

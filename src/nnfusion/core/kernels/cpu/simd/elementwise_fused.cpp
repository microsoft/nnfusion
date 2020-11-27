// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "elementwise_fused.hpp"

using namespace nnfusion;
using namespace nnfusion::kernels;
using namespace nnfusion::kernels::cpu;

DECLARE_int32(fthread_num_per_node);
int ElementwiseFused::unique_func_id = 0;

ElementwiseFused::ElementwiseFused(shared_ptr<KernelContext> ctx)
    : SimdKernelEmitter(ctx)
{
    NNFUSION_CHECK_NOT_NULLPTR(FuseContext());
}

std::shared_ptr<KernelContext> ElementwiseFused::FuseContext()
{
    std::shared_ptr<KernelContext> ctx = this->m_context;
    // output
    std::unordered_map<std::string, size_t> node_outputs;
    std::unordered_map<std::string, shared_ptr<nnfusion::descriptor::Tensor>> tensors;

    for (auto kernel_emitter : ctx->kernels)
    {
        auto gnode = kernel_emitter->m_context->gnode;
        for (size_t i = 0; i < gnode->get_input_size(); i++)
        {
            auto tv = gnode->get_input_tensor_ptr(i);
            NNFUSION_CHECK_NOT_NULLPTR(tv);
            auto iter = node_outputs.find(tv->get_name());
            if (iter == node_outputs.end())
            {
                ctx->inputs.push_back(tv);
                ctx->input_names.push_back(tv->get_name());
            }
            else
            {
                NNFUSION_CHECK(iter->second > 0);
                node_outputs[tv->get_name()] = node_outputs[tv->get_name()] - 1;
            }
        }

        for (size_t i = 0; i < gnode->get_output_size(); i++)
        {
            shared_ptr<descriptor::Tensor> tv = gnode->get_output_tensor_ptr(i);
            NNFUSION_CHECK_NOT_NULLPTR(tv);
            NNFUSION_CHECK(node_outputs.find(tv->get_name()) == node_outputs.end());
            NNFUSION_CHECK(gnode->get_output_users(i).size() > 0)
                << gnode->get_name() << " " << i << "th output has "
                << gnode->get_output_users(i).size() << " users.";
            node_outputs[tv->get_name()] = gnode->get_output_users(i).size();
            tensors.insert(std::make_pair(tv->get_name(), tv));
        }
    }

    for (auto& iter : node_outputs)
    {
        if (iter.second > 0)
        {
            ctx->output_names.push_back(iter.first);
            auto tw = tensors.find(iter.first);
            NNFUSION_CHECK(tw != tensors.end());
            ctx->outputs.push_back(tw->second);
        }
    }

    for (auto arg : ctx->inputs)
    {
        ctx->dtypes.push_back(arg->get_element_type().c_type_string());
    }

    for (auto out : ctx->outputs)
    {
        ctx->dtypes.push_back(out->get_element_type().c_type_string());
    }

    return ctx;
}

LanguageUnit_p ElementwiseFused::emit_function_body()
{
    create_ptr(LanguageUnit, lu_, get_function_name());
    LanguageUnit& lu = *lu_;

    bool has_not_float_elements = false;

    std::unordered_map<std::string, std::string> in_multi_data, in_single_data;
    for (int i = 0; i < m_context->inputs.size(); i++)
    {
        auto& tensor = m_context->inputs[i];
        in_args[tensor->get_name()] = "input" + std::to_string(i);
        if (tensor->get_element_type().c_type_string() != "float")
            has_not_float_elements = true;
    }
    for (int i = 0; i < m_context->outputs.size(); i++)
    {
        auto& tensor = m_context->outputs[i];
        out_args[tensor->get_name()] = "output" + std::to_string(i);
        out_types[tensor->get_name()] = tensor->get_element_type().c_type_string();
        if (tensor->get_element_type().c_type_string() != "float")
            has_not_float_elements = true;
    }

    for (auto kernel_emitter : m_context->kernels)
    {
        auto gnode = kernel_emitter->m_context->gnode;
        auto& out_tw = kernel_emitter->m_context->outputs[0];
        if (auto bc = std::dynamic_pointer_cast<nnfusion::op::Broadcast>(gnode->get_op_ptr()))
        {
            std::string op;
            if (bc->is_inner_broadcast())
            {
                op = " / " + std::to_string(bc->get_inner_broadcast_size());
            }
            else
            {
                NNFUSION_CHECK(bc->is_outer_broadcast());
                op = " % " + std::to_string(bc->get_outer_broadcast_size());
            }
            auto& in_tw = kernel_emitter->m_context->inputs[0];
            NNFUSION_CHECK(in_args.count(in_tw->get_name()) > 0);
            std::stringstream multi_data;
            multi_data << "__m256 simd_" << in_args[in_tw->get_name()] << " = _mm256_set_ps("
                       << in_args[in_tw->get_name()] << "[(i + 7)" << op << "], "
                       << in_args[in_tw->get_name()] << "[(i + 6)" << op << "], "
                       << in_args[in_tw->get_name()] << "[(i + 5)" << op << "], "
                       << in_args[in_tw->get_name()] << "[(i + 4)" << op << "], "
                       << in_args[in_tw->get_name()] << "[(i + 3)" << op << "], "
                       << in_args[in_tw->get_name()] << "[(i + 2)" << op << "], "
                       << in_args[in_tw->get_name()] << "[(i + 1)" << op << "], "
                       << in_args[in_tw->get_name()] << "[(i + 0)" << op << "]);";
            in_multi_data[in_args[in_tw->get_name()]] = multi_data.str();

            std::stringstream single_data;
            single_data << "__m256 simd_" << in_args[in_tw->get_name()]
                        << " = _mm256_insertf128_ps("
                        << "_mm256_setzero_ps(), _mm_set_ss(" << in_args[in_tw->get_name()] << "[i"
                        << op << "]), 0);";
            in_single_data[in_args[in_tw->get_name()]] = single_data.str();
        }
        else if (auto rs = std::dynamic_pointer_cast<nnfusion::op::Reshape>(gnode->get_op_ptr()))
        {
            NNFUSION_CHECK(rs->get_is_transpose() == false);
            auto& in_tw = kernel_emitter->m_context->inputs[0];
            if (in_args.count(in_tw->get_name()) > 0)
            {
                std::stringstream multi_data;
                if (in_tw->get_element_type().c_type_string() == "float")
                {
                    multi_data << "__m256 simd_" << in_args[in_tw->get_name()]
                               << " = _mm256_loadu_ps(" << in_args[in_tw->get_name()] << " + i);\n";
                }
                else
                {
                    multi_data << "for (int j = 0; j < " << m_simd_block_size << "; ++j)\n{\n";
                    multi_data << "tmp_buffer[j] = (float)(" << in_args[in_tw->get_name()]
                               << "[i + j]);\n}\n";
                    multi_data << "__m256 simd_" << in_args[in_tw->get_name()]
                               << " = _mm256_loadu_ps(tmp_buffer);\n";
                }
                in_multi_data[in_args[in_tw->get_name()]] = multi_data.str();

                std::stringstream single_data;
                single_data << "__m256 simd_" << in_args[in_tw->get_name()]
                            << " = _mm256_insertf128_ps("
                            << "_mm256_setzero_ps(), _mm_set_ss(" << in_args[in_tw->get_name()]
                            << "[i]), 0);";
                in_single_data[in_args[in_tw->get_name()]] = single_data.str();

                in_args[out_tw->get_name()] = in_args[in_tw->get_name()];
            }
        }
        else
        {
            auto simd_kernel = std::dynamic_pointer_cast<SimdKernelEmitter>(kernel_emitter);
            NNFUSION_CHECK_NOT_NULLPTR(simd_kernel)
                << "kernel type:" << kernel_emitter->m_context->gnode->get_op_type();

            for (int i = 0; i < simd_kernel->m_context->inputs.size(); i++)
            {
                auto& in_tw = simd_kernel->m_context->inputs[i];
                if (in_args.count(in_tw->get_name()) > 0)
                {
                    std::stringstream multi_data;
                    if (in_tw->get_element_type().c_type_string() == "float")
                    {
                        multi_data << "__m256 simd_" << in_args[in_tw->get_name()]
                                   << " = _mm256_loadu_ps(" << in_args[in_tw->get_name()]
                                   << " + i);";
                    }
                    else
                    {
                        multi_data << "for (int j = 0; j < " << m_simd_block_size << "; ++j)\n{\n";
                        multi_data << "tmp_buffer[j] = (float)(" << in_args[in_tw->get_name()]
                                   << "[i + j]);\n}\n";
                        multi_data << "__m256 simd_" << in_args[in_tw->get_name()]
                                   << " = _mm256_loadu_ps(tmp_buffer);";
                    }
                    in_multi_data[in_args[in_tw->get_name()]] = multi_data.str();

                    std::stringstream single_data;
                    single_data << "__m256 simd_" << in_args[in_tw->get_name()]
                                << " = _mm256_insertf128_ps("
                                << "_mm256_setzero_ps(), _mm_set_ss(" << in_args[in_tw->get_name()]
                                << "[i]), 0);";
                    in_single_data[in_args[in_tw->get_name()]] = single_data.str();
                }
            }
        }
    }

    size_t data_size = 0;
    for (auto out : m_context->outputs)
    {
        auto size = static_cast<uint32_t>(nnfusion::shape_size(out->get_shape()));
        if (size > data_size)
            data_size = size;
    }
    size_t remainder_count = data_size % m_simd_block_size;
    size_t loop_count = data_size - remainder_count;
    size_t shard_data_count = data_size / m_simd_block_size;

    if (loop_count > 0)
    {
        lu << "const int64_t min_cost_per_shard = 10000;\n";
        lu << "int num_shards = "
           << "std::max(std::min(static_cast<int64_t>("
           << "thread_pool->NumThreads()), " << loop_count
           << " / min_cost_per_shard), static_cast<int64_t>(1));\n";
        lu << "const int64_t block_size = (" << shard_data_count
           << " + num_shards - 1) / num_shards;\n";
        lu << "if (block_size > " << shard_data_count << ")\n";
        lu.block_begin();
        lu << "num_shards = 1;\n";
        lu.block_end();

        if (has_not_float_elements)
        {
            lu << "float tmp_buffer[" << m_simd_block_size << "];\n";
        }

        lu << "auto func = [&](int __rank__)\n";
        lu << "{\n";
        lu << "int64_t start = block_size * __rank__ * " << m_simd_block_size << ";\n";
        lu << "int64_t end = std::min(block_size * (__rank__ + 1), static_cast<int64_t>("
           << shard_data_count << ")) * " << m_simd_block_size << ";\n";

        lu << "for (size_t i = start; i < end; i+=" << m_simd_block_size << ")\n";
        lu.block_begin();
        for (const auto& simd_init : in_multi_data)
        {
            lu << simd_init.second << "\n";
        }
        FuseFunctionBody(lu);

        for (auto& pair : out_args)
        {
            string out_string = "";
            if (local_tensors.count(pair.first) > 0)
            {
                out_string = local_tensors[pair.first];
            }
            else
            {
                NNFUSION_CHECK(in_args.count(pair.first) > 0);
                out_string = "simd_" + in_args[pair.first];
            }
            if (out_types[pair.first] == "float")
            {
                lu << "_mm256_storeu_ps(" << pair.second << " + i, ";
                lu << out_string << ");\n";
            }
            else
            {
                lu << "_mm256_storeu_ps(tmp_buffer, " << out_string << ");\n";
                lu << "for (int j = 0; j < " << m_simd_block_size << "; ++j)\n{\n";
                lu << pair.second << "[i + j] = (" << out_types[pair.first]
                   << ")tmp_buffer[j];\n}\n";
            }
        }
        lu.block_end();
        lu << "};\n";
        lu << "thread_pool->ParallelFor(num_shards, func);\n";
    }

    if (remainder_count > 0)
    {
        lu << "for (size_t i = " << loop_count << "; i < " << data_size << "; ++i)\n";
        lu << "{\n";

        for (const auto& simd_init : in_single_data)
        {
            lu << simd_init.second << "\n";
        }
        FuseFunctionBody(lu);

        for (auto& pair : out_args)
        {
            lu << pair.second << "[i] = _mm_cvtss_f32(_mm256_extractf128_ps(";
            if (local_tensors.count(pair.first) > 0)
            {
                lu << local_tensors[pair.first] << ", 0));\n";
            }
            else
            {
                NNFUSION_CHECK(in_args.count(pair.first) > 0);
                lu << "simd_" << in_args[pair.first] << ", 0));\n";
            }
        }
        lu << "}\n";
    }

    return lu_;
}

void ElementwiseFused::FuseFunctionBody(LanguageUnit& lu)
{
    size_t temp_tensor_id = 0;
    for (auto kernel_emitter : m_context->kernels)
    {
        auto gnode = kernel_emitter->m_context->gnode;
        auto& out_tw = kernel_emitter->m_context->outputs[0];
        if (auto bc = std::dynamic_pointer_cast<nnfusion::op::Broadcast>(gnode->get_op_ptr()))
        {
            local_tensors[out_tw->get_name()] = "temp" + std::to_string(temp_tensor_id++);
            auto& in_tw = kernel_emitter->m_context->inputs[0];
            NNFUSION_CHECK(in_args.count(in_tw->get_name()) > 0);

            lu << "__m256 " << local_tensors[out_tw->get_name()] << " = simd_"
               << in_args[in_tw->get_name()] << ";\n";
        }
        else if (auto rs = std::dynamic_pointer_cast<nnfusion::op::Reshape>(gnode->get_op_ptr()))
        {
            NNFUSION_CHECK(rs->get_is_transpose() == false);
            auto& in_tw = kernel_emitter->m_context->inputs[0];
            if (in_args.count(in_tw->get_name()) > 0)
            {
                in_args[out_tw->get_name()] = in_args[in_tw->get_name()];
            }
            else
            {
                NNFUSION_CHECK(local_tensors.count(in_tw->get_name()) > 0);
                local_tensors[out_tw->get_name()] = local_tensors[in_tw->get_name()];
            }
        }
        else
        {
            auto simd_kernel = std::dynamic_pointer_cast<SimdKernelEmitter>(kernel_emitter);
            NNFUSION_CHECK_NOT_NULLPTR(simd_kernel)
                << "kernel type:" << kernel_emitter->m_context->gnode->get_op_type();
            auto op_kernel = simd_kernel->get_op_kernel();
            if (op_kernel.second != nullptr)
            {
                lu.require(op_kernel.second);
            }
            local_tensors[out_tw->get_name()] = "temp" + std::to_string(temp_tensor_id++);
            std::vector<std::string> input_args;
            for (int i = 0; i < simd_kernel->m_context->inputs.size(); i++)
            {
                auto& in_tw = simd_kernel->m_context->inputs[i];
                if (in_args.count(in_tw->get_name()) > 0)
                {
                    input_args.push_back("simd_" + in_args[in_tw->get_name()]);
                }
                else
                {
                    NNFUSION_CHECK(local_tensors.count(in_tw->get_name()) > 0);
                    input_args.push_back(local_tensors[in_tw->get_name()]);
                }
            }
            lu << "__m256 " << local_tensors[out_tw->get_name()] << " = " << op_kernel.first << "("
               << join(input_args, ", ") << ");\n";
        }
    }
}

LanguageUnit_p ElementwiseFused::emit_dependency()
{
    LanguageUnit_p _lu(new LanguageUnit(get_function_name() + "_dep"));
    _lu->require(header::simd);

    return _lu;
}

LanguageUnit_p ElementwiseFused::emit_function_name()
{
    LanguageUnit_p _lu(new LanguageUnit("function_name"));
    auto& lu = *_lu;

    std::vector<std::string> names;
    for (auto kernel : m_context->kernels)
    {
        names.push_back(kernel->m_context->gnode->get_op_type());
    }

    lu << "FusedKernel_" << join(m_context->dtypes, "_") << "_" << m_kernel_type << "_"
       << join(names, "_") << "_" << ElementwiseFused::unique_func_id++; //<< custom_tag;

    return _lu;
}

LanguageUnit_p ElementwiseFused::emit_comments()
{
    LanguageUnit_p _lu(new LanguageUnit(this->m_kernel_name + "_comments"));
    auto& lu = *_lu;
    lu << "// Node name:\t Elementwise Kernel Fusion"
       << "\n";
    //lu << "// Description:\t" << m_context->node->description() << "\n";
    lu << "// Input:\n";
    for (auto in : m_context->inputs)
    {
        lu << "//\t- name: " << in->get_name();
        lu << "\ttype: " << in->get_element_type().c_type_string();
        lu << "\tshape: " << in->get_shape();
        lu << "\n";
    }

    lu << "// Output:\n";
    for (auto out : m_context->outputs)
    {
        lu << "//\t- name: " << out->get_name();
        lu << "\ttype: " << out->get_element_type().c_type_string();
        lu << "\tshape: " << out->get_shape();
        lu << "\n";
    }

    lu << "// Fused functions:\n";
    for (auto kernel : m_context->kernels)
    {
        lu << "// " << kernel->get_or_emit_source()->name_unit->get_code()
           << kernel->get_or_emit_source()->call_unit->get_code();
    }

    return _lu;
}

REGISTER_KERNEL_EMITTER(
    "ElementwiseFused",                                                       // op_name
    Device(GENERIC_CPU).TypeConstraint(element::f32).Tag("simd").Priority(5), // attrs
    cpu::ElementwiseFused)

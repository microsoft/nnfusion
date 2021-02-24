// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "elementwise_fused.hpp"
#include "elementwise.hpp"

using namespace nnfusion;
using namespace nnfusion::kernels;
using namespace nnfusion::kernels::cuda;

int ElementWiseFused::unique_func_id = 0;

ElementWiseFused::ElementWiseFused(shared_ptr<KernelContext> ctx)
    : BlockCudaEmitter(ctx)
{
    NNFUSION_CHECK_NOT_NULLPTR(FuseContext());
}

std::shared_ptr<KernelContext> ElementWiseFused::FuseContext()
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

std::pair<std::string, shared_ptr<LanguageUnit>> get_op_kernel(shared_ptr<KernelContext> ctx)
{
    auto iter = CudaElementOpMap.find(ctx->gnode->get_op_type());
    NNFUSION_CHECK(iter != CudaElementOpMap.end()) << "unable find op type: "
                                                   << ctx->gnode->get_op_type();
    std::string op = iter->second.op;
    shared_ptr<LanguageUnit> kernel = nullptr;

    if (iter->second.math_kernel != "")
    {
        std::vector<std::string> data_types;
        for (auto arg : ctx->inputs)
        {
            data_types.push_back(arg->get_element_type().c_type_string());
        }
        data_types.push_back(ctx->outputs[0]->get_element_type().c_type_string());
        kernel = get_math_kernel(op, iter->second.math_kernel, data_types);
        NNFUSION_CHECK_NOT_NULLPTR(kernel);
    }
    return std::make_pair(op, kernel);
}

LanguageUnit_p ElementWiseFused::emit_function_body()
{
    create_ptr(LanguageUnit, lu_, get_function_name());
    LanguageUnit& lu = *lu_;

    std::unordered_map<std::string, std::string> in_args, out_args, local_tensors;
    for (int i = 0; i < m_context->inputs.size(); i++)
    {
        auto& tensor = m_context->inputs[i];
        in_args[tensor->get_name()] = "input" + std::to_string(i);
    }
    for (int i = 0; i < m_context->outputs.size(); i++)
    {
        auto& tensor = m_context->outputs[i];
        out_args[tensor->get_name()] = "output" + std::to_string(i);
    }

    size_t temp_tensor_id = 0;

    uint32_t nthreads = 0;
    for (auto out : m_context->outputs)
    {
        auto size = static_cast<uint32_t>(nnfusion::shape_size(out->get_shape()));
        if (size > nthreads)
            nthreads = size;
    }

    int grids, blocks, bound;
    compute_best_config(grids, blocks, bound);

    if (grids == 1)
    {
        lu << "int tid = threadIdx.x;\n";
    }
    else
    {
        lu << "int tid = blockIdx.x * " << std::to_string(blocks) << " + threadIdx.x;\n";
    }
    if (bound)
    {
        lu << "if (tid >= " << bound << ") return;\n";
    }

    for (auto kernel_emitter : m_context->kernels)
    {
        auto gnode = kernel_emitter->m_context->gnode;
        auto& out_tw = kernel_emitter->m_context->outputs[0];
        if (auto bc = std::dynamic_pointer_cast<nnfusion::op::Broadcast>(gnode->get_op_ptr()))
        {
            std::string index = "";
            if (bc->is_inner_broadcast())
            {
                index += "[tid / " + std::to_string(bc->get_inner_broadcast_size()) + "]";
            }
            else
            {
                NNFUSION_CHECK(bc->is_outer_broadcast());
                index += "[tid % " + std::to_string(bc->get_outer_broadcast_size()) + "]";
            }
            local_tensors[out_tw->get_name()] = "temp" + std::to_string(temp_tensor_id++);
            auto& in_tw = kernel_emitter->m_context->inputs[0];
            NNFUSION_CHECK(in_args.count(in_tw->get_name()) > 0);

            lu << out_tw->get_element_type().c_type_string() << " "
               << local_tensors[out_tw->get_name()] << " = " << in_args[in_tw->get_name()] << index
               << ";\n";
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
            auto cuda_kernel = std::dynamic_pointer_cast<CudaElementwiseEmitter>(kernel_emitter);

            if (!cuda_kernel)
            {
                bool check_flag = false;

                auto ir_kernel =
                    std::dynamic_pointer_cast<AntaresCudaKernelEmitter>(kernel_emitter);
                if (ir_kernel &&
                    CudaElementOpMap.find(ir_kernel->m_context->gnode->get_op_type()) !=
                        CudaElementOpMap.end())
                {
                    check_flag = true;
                }

                auto cache_kernel = std::dynamic_pointer_cast<CacheCudaEmitter>(kernel_emitter);
                if (cache_kernel &&
                    CudaElementOpMap.find(cache_kernel->m_context->gnode->get_op_type()) !=
                        CudaElementOpMap.end())
                {
                    check_flag = true;
                }

                auto cache_block_kernel =
                    std::dynamic_pointer_cast<CacheBlockCudaEmitter>(kernel_emitter);
                if (cache_block_kernel &&
                    CudaElementOpMap.find(cache_block_kernel->m_context->gnode->get_op_type()) !=
                        CudaElementOpMap.end())
                {
                    check_flag = true;
                }

                if (!check_flag)
                {
                    NNFUSION_CHECK_FAIL() << "Illegal element-wise kernel: "
                                          << kernel_emitter->m_context->gnode->get_op_type();
                }
            }

            std::string invoke_func;
            if (kernel_emitter->m_context->gnode->get_op_type() == "Convert")
            {
                lu.require(declaration::cuda_convert_template);
                lu.require(header::cublas);
                invoke_func =
                    "convert<" +
                    kernel_emitter->m_context->inputs[0]->get_element_type().c_type_string() +
                    ", " +
                    kernel_emitter->m_context->outputs[0]->get_element_type().c_type_string() + ">";
            }
            else
            {
                auto op_kernel = get_op_kernel(kernel_emitter->m_context);
                if (op_kernel.second != nullptr)
                {
                    lu.require(op_kernel.second);
                    if (kernel_emitter->m_context->gnode->get_op_type() == "Gelu")
                    {
                        op_kernel.second->require(declaration::math_Gelu);
                        op_kernel.second->require(header::cublas);
                    }
                }
                invoke_func = op_kernel.first;
            }
            local_tensors[out_tw->get_name()] = "temp" + std::to_string(temp_tensor_id++);
            std::vector<std::string> input_args;
            for (int i = 0; i < kernel_emitter->m_context->inputs.size(); i++)
            {
                auto& in_tw = kernel_emitter->m_context->inputs[i];
                if (in_args.count(in_tw->get_name()) > 0)
                {
                    input_args.push_back(in_args[in_tw->get_name()] + "[tid]");
                }
                else
                {
                    NNFUSION_CHECK(local_tensors.count(in_tw->get_name()) > 0);
                    input_args.push_back(local_tensors[in_tw->get_name()]);
                }
            }
            lu << out_tw->get_element_type().c_type_string() << " "
               << local_tensors[out_tw->get_name()] << " = " << invoke_func << "("
               << join(input_args, ", ") << ");\n";
        }
    }

    for (auto& pair : out_args)
    {
        lu << pair.second << "[tid] = ";
        if (local_tensors.count(pair.first) > 0)
        {
            lu << local_tensors[pair.first] << ";\n";
        }
        else
        {
            NNFUSION_CHECK(in_args.count(pair.first) > 0);
            lu << in_args[pair.first] << "[tid];\n";
        }
    }

    return lu_;
}

LanguageUnit_p ElementWiseFused::emit_dependency()
{
    LanguageUnit_p _lu(new LanguageUnit(get_function_name() + "_dep"));
    _lu->require(header::cuda);
    _lu->require(header::stdio);

    return _lu;
}

LanguageUnit_p ElementWiseFused::emit_function_name()
{
    LanguageUnit_p _lu(new LanguageUnit("function_name"));
    auto& lu = *_lu;

    std::vector<std::string> names;
    for (auto kernel : m_context->kernels)
    {
        names.push_back(kernel->m_context->gnode->get_op_type());
    }

    lu << "FusedKernel_" << join(m_context->dtypes, "_") << "_" << m_kernel_type << "_"
       << join(names, "_") << "_" << ElementWiseFused::unique_func_id++; //<< custom_tag;

    return _lu;
}

LanguageUnit_p ElementWiseFused::emit_comments()
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

void ElementWiseFused::set_launch_config()
{
    int grids, blocks, bound;
    compute_best_config(grids, blocks, bound);

    m_gridDim = dim3(grids, 1, 1);
    m_blockDim = dim3(blocks, 1, 1);
}

void ElementWiseFused::compute_best_config(int& grids, int& blocks, int& bound)
{
    uint32_t num_ele =
        static_cast<uint32_t>(nnfusion::shape_size(m_context->outputs[0]->get_shape()));
    for (int i = 512; i >= 64; i >>= 1)
    {
        if (num_ele % i == 0)
        {
            grids = num_ele / i, blocks = i, bound = 0;
            return;
        }
    }
    for (int i = 512; i >= 32; i--)
    {
        if (num_ele % i == 0)
        {
            grids = num_ele / i, blocks = i, bound = 0;
            return;
        }
    }
    if (num_ele < 32)
        grids = 1, blocks = num_ele, bound = 0;
    else
        grids = (num_ele + 255) / 256, blocks = 256, bound = 1;
}

REGISTER_KERNEL_EMITTER(
    "ElementWiseFused",                                                           // op_name
    Device(CUDA_GPU).TypeConstraint(element::f32).Tag("cuda_kernel").Priority(2), // attrs
    cuda::ElementWiseFused)
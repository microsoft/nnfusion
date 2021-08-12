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
    m_gnode = static_pointer_cast<graph::FusedGNode>(ctx->gnode);
    NNFUSION_CHECK_NOT_NULLPTR(m_gnode);
}

std::pair<std::string, shared_ptr<LanguageUnit>> get_op_kernel(shared_ptr<graph::OpContext> ctx)
{
    auto iter = CudaElementOpMap.find(ctx->op->get_op_type());
    NNFUSION_CHECK(iter != CudaElementOpMap.end()) << "unable find op type: "
                                                   << ctx->op->get_op_type();
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

    for (auto op_ctx : m_gnode->get_op_contexts())
    {
        auto& out_tw = op_ctx->outputs[0];
        if (auto bc = std::dynamic_pointer_cast<nnfusion::op::Broadcast>(op_ctx->op))
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
            auto& in_tw = op_ctx->inputs[0];
            NNFUSION_CHECK(in_args.count(in_tw->get_name()) > 0);

            lu << out_tw->get_element_type().c_type_string() << " "
               << local_tensors[out_tw->get_name()] << " = " << in_args[in_tw->get_name()] << index
               << ";\n";
        }
        else if (auto rs = std::dynamic_pointer_cast<nnfusion::op::Reshape>(op_ctx->op))
        {
            NNFUSION_CHECK(rs->get_is_transpose() == false);
            auto& in_tw = op_ctx->inputs[0];
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
            if (CudaElementOpMap.find(op_ctx->op->get_op_type()) == CudaElementOpMap.end())
            {
                NNFUSION_CHECK_FAIL() << "Illegal element-wise kernel: "
                                      << op_ctx->op->get_op_type();
            }

            std::string invoke_func;
            if (op_ctx->op->get_op_type() == "Convert")
            {
                lu.require(declaration::cuda_convert_template);
                lu.require(header::cublas);
                invoke_func = "convert<" + op_ctx->inputs[0]->get_element_type().c_type_string() +
                              ", " + op_ctx->outputs[0]->get_element_type().c_type_string() + ">";
            }
            else
            {
                auto op_kernel = get_op_kernel(op_ctx);
                if (op_kernel.second != nullptr)
                {
                    lu.require(op_kernel.second);
                    if (op_ctx->op->get_op_type() == "Gelu")
                    {
                        op_kernel.second->require(declaration::math_Gelu);
                        op_kernel.second->require(header::cublas);
                    }
                }
                invoke_func = op_kernel.first;
            }
            local_tensors[out_tw->get_name()] = "temp" + std::to_string(temp_tensor_id++);
            std::vector<std::string> input_args;
            for (int i = 0; i < op_ctx->inputs.size(); i++)
            {
                auto& in_tw = op_ctx->inputs[i];
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
            NNFUSION_CHECK(in_args.count(pair.first) > 0) << m_context->gnode->get_name() << " "
                                                          << lu.get_code() << " " << pair.first;
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
    for (auto ctx : m_gnode->get_op_contexts())
    {
        names.push_back(ctx->op->get_op_type());
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
    for (auto ctx : m_gnode->get_op_contexts())
    {
        lu << "// " << ctx->op->get_op_type() << ", " << ctx->op->get_name() << "\n";
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
// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "recursion.hpp"
#include "../cuda_cudnn.hpp"
#include "convolution.hpp"
#include "nnfusion/core/operators/op_define/recursion.hpp"
#include "nnfusion/engine/pass/graph/blockfusion/blockfusion_codegen.hpp"

using namespace nnfusion;
using namespace nnfusion::kernels;

cuda::FuncForward::FuncForward(shared_ptr<KernelContext> ctx)
    : CudaEmitter(ctx)
{
    std::stringstream tag;
    tag << "_FuncForwardOP";
    custom_tag = tag.str();
    auto m_workspace = allocate_tensor(Shape{0}, nnfusion::element::character);
    m_workspace->set_name("recursion_stack");
    m_context->inputs.push_back(m_workspace);
    m_context->input_names.push_back(m_workspace->get_name());
}

LanguageUnit_p cuda::FuncForward::emit_function_body()
{
    LanguageUnit_p _lu(new LanguageUnit(get_function_name()));
    return _lu;
}

LanguageUnit_p cuda::FuncForward::emit_dependency()
{
    LanguageUnit_p _lu(new LanguageUnit(get_function_name() + "_dep"));
    return _lu;
}

void cuda::FuncForward::set_launch_config()
{
}

LanguageUnit_p cuda::FuncForward::emit_block_kernel_call(std::vector<std::string> params)
{
    LanguageUnit_p _lu(new LanguageUnit(this->m_kernel_name + "_device_kernel_call"));
    auto& lu = *_lu;
    if (m_block_func_name == "")
        m_block_func_name = m_kernel_name + "_recursion";
    params.push_back("shared_buffer");

    lu << m_block_func_name + "_block_kernel(" << join(params, ", ") << ");\n";
    return _lu;
}

std::string cuda::FuncForward::m_block_func_name = "";

cuda::Recursion::Recursion(shared_ptr<KernelContext> ctx)
    : ControlFlowEmitter(ctx)
{
    std::stringstream tag;
    tag << "_RecursionOP";
    custom_tag = tag.str();
    auto op = static_pointer_cast<op::Recursion>(ctx->gnode->get_op_ptr());
    NNFUSION_CHECK_NOT_NULLPTR(op);
    m_loop_body_tu = op->get_body_tu();
    m_workspace_size = 0;
    for (auto& pair : m_loop_body_tu->memory_allocator_factory->get_allocator_list())
    {
        m_workspace_size += pair.second->max_allocated();
    }
    m_workspace = allocate_tensor(Shape{m_workspace_size * 20}, nnfusion::element::character);
    m_context->inputs.push_back(m_workspace);
    m_context->input_names.push_back(m_workspace->get_name());
    m_loop_output_map = op->get_output_map();
    m_block_func_name = move(FuncForward::m_block_func_name);
    NNFUSION_CHECK(!m_block_func_name.empty());
    m_shared_memory_size = get_subgraph_shared_memory(m_loop_body_tu->program);
    bypass_instructions(m_loop_body_tu->program);
}

void cuda::Recursion::generate_subgraph_code(LanguageUnit_p _lu)
{
    auto& lu = *_lu;
    auto instructions = get_fused_kernel(m_loop_body_tu->program);
    auto inputs = get_subgraph_inputs(m_loop_body_tu->program);
    for (auto ins : instructions)
    {
        auto kernel = static_pointer_cast<cuda::CudaEmitter>(ins->getKernel());
        lu << get_launch_bound(ins);
        std::vector<string> params;
        for (auto tensor : ins->get_inputs())
        {
            if (inputs.count(tensor->get_name()))
                params.push_back("input" + std::to_string(inputs[tensor->get_name()]));
            else
                params.push_back(get_workspace_tensor(tensor));
        }
        for (auto tensor : ins->get_outputs())
        {
            if (m_loop_output_map.count(tensor->get_name(false)))
            {
                auto output_index = m_loop_output_map[tensor->get_name(false)];
                params.push_back("output" + std::to_string(output_index));
            }
            else
                params.push_back(get_workspace_tensor(tensor));
        }
        if (std::dynamic_pointer_cast<BlockFusionCudaCodegen>(kernel) != nullptr)
            for (auto tensor : kernel->m_context->tensors)
                params.push_back(get_workspace_tensor(tensor));
        lu << kernel->emit_block_kernel_call(params)->get_code();
        if (ins != instructions.back())
            lu << "Barrier();\n";
    }
}

LanguageUnit_p cuda::Recursion::emit_function_body()
{
    LanguageUnit_p _lu(new LanguageUnit(get_function_name()));
    auto& lu = *_lu;
    allocate_shared_memory(_lu);
    generate_subgraph_code(_lu);
    return _lu;
}

void cuda::Recursion::set_launch_config()
{
    auto cfg0 = get_subgraph_launch_config(m_loop_body_tu->program);
    m_blockDim = cfg0.first;
    m_gridDim = cfg0.second;
}

static void replace_dep_code(LanguageUnit_p lu, const std::string& src, const std::string& tgt)
{
    auto code = lu->get_code();
    lu->code_symbol_replace(src, tgt);
    for (auto item : lu->local_symbol)
        replace_dep_code(item.second, src, tgt);
}

LanguageUnit_p cuda::Recursion::emit_dependency()
{
    LanguageUnit_p _lu(new LanguageUnit(get_function_name() + "_dep"));
    _lu->require(header::cuda);
    _lu->require(declaration::barrier);
    auto saved = m_kernel_name;
    m_kernel_name = m_block_func_name;
    // include the recursion kernel declare at first
    auto kernel_declare = this->emit_device_function_signature();
    (*kernel_declare) << ";\n";
    _lu->require(kernel_declare);
    LanguageUnit_p kernel_code(new LanguageUnit(get_function_name() + "_block_kernel"));
    (*kernel_code) << this->emit_device_function_signature()->get_code();
    kernel_code->block_begin();
    is_emitting_block_kernel = true;
    (*kernel_code) << emit_function_body()->get_code();
    is_emitting_block_kernel = false;
    kernel_code->block_end();
    m_kernel_name = saved;

    for (auto ins : get_fused_kernel(m_loop_body_tu->program))
    {
        auto kernel = static_pointer_cast<cuda::CudaEmitter>(ins->getKernel());
        auto body = kernel->get_or_emit_source();
        auto block_kernel = kernel->emit_block_kernel();
        block_kernel->require(body->dep_unit);
        _lu->require(block_kernel);
        kernel_code->require(block_kernel);
    }

    _lu->require(kernel_code);
    replace_dep_code(_lu, "$stack_size$", std::to_string(m_workspace_size));
    return _lu;
}

REGISTER_KERNEL_EMITTER("FuncForward",                                             // op_name
                        Device(CUDA_GPU).TypeConstraint(element::f32).Priority(2), // attrs
                        cuda::FuncForward)
REGISTER_KERNEL_EMITTER("Recursion",                                               // op_name
                        Device(CUDA_GPU).TypeConstraint(element::f32).Priority(2), // attrs
                        cuda::Recursion)

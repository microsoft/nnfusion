// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "if.hpp"
#include "../cuda_cudnn.hpp"
#include "convolution.hpp"
#include "nnfusion/core/operators/op_define/if.hpp"
#include "nnfusion/engine/pass/graph/blockfusion/blockfusion_codegen.hpp"

using namespace nnfusion;
using namespace nnfusion::kernels;

cuda::If::If(shared_ptr<KernelContext> ctx)
    : ControlFlowEmitter(ctx)
{
    std::stringstream tag;
    tag << "_IfOP";
    custom_tag = tag.str();
    auto op = static_pointer_cast<op::If>(ctx->gnode->get_op_ptr());
    NNFUSION_CHECK_NOT_NULLPTR(op);
    m_then_branch_tu = op->get_then_branch_tu();
    m_else_branch_tu = op->get_else_branch_tu();
    size_t size0 = 0, size1 = 0;
    for (auto& pair : m_then_branch_tu->memory_allocator_factory->get_allocator_list())
        size0 += pair.second->max_allocated();
    for (auto& pair : m_else_branch_tu->memory_allocator_factory->get_allocator_list())
        size1 += pair.second->max_allocated();
    m_workspace = allocate_tensor(Shape({std::max(size0, size1)}), nnfusion::element::character);
    m_context->inputs.push_back(m_workspace);
    m_context->input_names.push_back(m_workspace->get_name());
    m_output_map = op->get_output_map();
}

void cuda::If::generate_branch_code(LanguageUnit_p _lu, bool else_branch = false)
{
    auto tu = m_then_branch_tu;
    if (else_branch)
    {
        tu = m_else_branch_tu;
    }
    auto& lu = *_lu;
    auto instructions = get_fused_kernel(tu->program);
    auto inputs = get_subgraph_inputs(tu->program);
    for (auto ins : instructions)
    {
        auto kernel = static_pointer_cast<cuda::CudaEmitter>(ins->getKernel());
        lu << get_launch_bound(ins);
        std::vector<string> params;
        int tensor_cnt = 0;
        for (auto tensor : ins->get_inputs())
        {
            if (inputs.count(tensor->get_name()))
            {
                auto input_index = inputs[tensor->get_name()];
                params.push_back("input" + std::to_string(input_index));
            }
            else
                params.push_back(get_workspace_tensor(tensor));
        }
        for (auto tensor : ins->get_outputs())
        {
            if (m_output_map.count(tensor->get_name(false)))
                params.push_back("output" + std::to_string(m_output_map[tensor->get_name(false)]));
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

LanguageUnit_p cuda::If::emit_function_body()
{
    LanguageUnit_p _lu(new LanguageUnit(get_function_name()));
    auto& lu = *_lu;
    lu << "if (*input0) ";
    lu.block_begin();
    generate_branch_code(_lu, false);
    lu.block_end();
    lu << "else ";
    lu.block_begin();
    generate_branch_code(_lu, true);
    lu.block_end();
    return _lu;
}

void cuda::If::set_launch_config()
{
    auto cfg0 = get_subgraph_launch_config(m_then_branch_tu->program);
    auto cfg1 = get_subgraph_launch_config(m_else_branch_tu->program);
    m_blockDim = maxdim3(cfg0.first, cfg1.first);
    m_gridDim = maxdim3(cfg0.second, cfg1.second);
}

LanguageUnit_p cuda::If::emit_dependency()
{
    LanguageUnit_p _lu(new LanguageUnit(get_function_name() + "_dep"));
    _lu->require(header::cuda);
    _lu->require(declaration::barrier);
    for (auto ins : get_fused_kernel(m_then_branch_tu->program))
    {
        auto kernel = static_pointer_cast<cuda::CudaEmitter>(ins->getKernel());
        auto body = kernel->get_or_emit_source();
        auto block_kernel = kernel->emit_block_kernel();
        block_kernel->require(body->dep_unit);
        _lu->require(block_kernel);
    }
    for (auto ins : get_fused_kernel(m_else_branch_tu->program))
    {
        auto kernel = static_pointer_cast<cuda::CudaEmitter>(ins->getKernel());
        auto body = kernel->get_or_emit_source();
        auto block_kernel = kernel->emit_block_kernel();
        block_kernel->require(body->dep_unit);
        _lu->require(block_kernel);
    }
    return _lu;
}

REGISTER_KERNEL_EMITTER("If",                                                      // op_name
                        Device(CUDA_GPU).TypeConstraint(element::f32).Priority(2), // attrs
                        cuda::If)

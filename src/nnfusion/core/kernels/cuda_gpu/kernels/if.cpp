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
    std::unordered_map<std::string, size_t> then_branch_pool_offset, else_branch_pool_offset;
    for (auto& pair : m_then_branch_tu->memory_allocator_factory->get_allocator_list())
    {
        then_branch_pool_offset[pair.second->get_name()] = size0;
        size0 += pair.second->max_allocated();
    }
    for (auto& pair : m_else_branch_tu->memory_allocator_factory->get_allocator_list())
    {
        else_branch_pool_offset[pair.second->get_name()] = size1;
        size1 += pair.second->max_allocated();
    }
    m_workspace = allocate_tensor(Shape({std::max(size0, size1)}), nnfusion::element::character);
    m_context->inputs.push_back(m_workspace);
    m_context->input_names.push_back(m_workspace->get_name());
    m_shared_memory_size = max(get_subgraph_shared_memory(m_then_branch_tu->program),
                               get_subgraph_shared_memory(m_else_branch_tu->program));
    m_pool_offset = then_branch_pool_offset;
    m_then_branch_instructions = create_param_map(m_then_branch_tu->program, op->get_output_map());
    m_pool_offset = else_branch_pool_offset;
    m_else_branch_instructions = create_param_map(m_else_branch_tu->program, op->get_output_map());
}

void cuda::If::generate_branch_code(LanguageUnit_p _lu, bool else_branch = false)
{
    auto instructions = m_then_branch_instructions;
    if (else_branch)
        instructions = m_else_branch_instructions;
    auto& lu = *_lu;
    for (auto ins : *instructions)
    {
        auto kernel = static_pointer_cast<cuda::CudaEmitter>(ins->getKernel());
        lu << get_launch_bound(ins);
        std::vector<string> params;
        for (auto tensor : ins->get_inputs())
            params.push_back(m_param_map[tensor]);
        for (auto tensor : ins->get_outputs())
            params.push_back(m_param_map[tensor]);
        if (std::dynamic_pointer_cast<BlockFusionCudaCodegen>(kernel) != nullptr)
            for (auto tensor : kernel->m_context->tensors)
                params.push_back(m_param_map[tensor]);
        lu << kernel->emit_block_kernel_call(params)->get_code();
        if (ins != instructions->back())
            lu << "Barrier();\n";
    }
}

LanguageUnit_p cuda::If::emit_function_body()
{
    LanguageUnit_p _lu(new LanguageUnit(get_function_name()));
    auto& lu = *_lu;
    allocate_shared_memory(_lu);
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
    auto cfg0 = get_subgraph_launch_config(m_then_branch_instructions);
    auto cfg1 = get_subgraph_launch_config(m_else_branch_instructions);
    m_blockDim = maxdim3(cfg0.first, cfg1.first);
    m_gridDim = maxdim3(cfg0.second, cfg1.second);
}

LanguageUnit_p cuda::If::emit_dependency()
{
    LanguageUnit_p _lu(new LanguageUnit(get_function_name() + "_dep"));
    _lu->require(header::cuda);
    _lu->require(declaration::barrier);
    for (auto ins : *m_then_branch_instructions)
    {
        auto kernel = static_pointer_cast<cuda::CudaEmitter>(ins->getKernel());
        auto body = kernel->get_or_emit_source();
        auto block_kernel = kernel->emit_block_kernel();
        block_kernel->require(body->dep_unit);
        _lu->require(block_kernel);
    }
    for (auto ins : *m_else_branch_instructions)
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

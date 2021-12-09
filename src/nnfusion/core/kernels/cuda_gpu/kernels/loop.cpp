// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "loop.hpp"
#include "../cuda_cudnn.hpp"
#include "convolution.hpp"
#include "nnfusion/core/operators/op_define/loop.hpp"
#include "nnfusion/engine/pass/graph/blockfusion/blockfusion_codegen.hpp"

using namespace nnfusion;
using namespace nnfusion::kernels;

cuda::Loop::Loop(shared_ptr<KernelContext> ctx)
    : ControlFlowEmitter(ctx)
{
    std::stringstream tag;
    tag << "_LoopOP";
    custom_tag = tag.str();
    auto op = static_pointer_cast<op::Loop>(ctx->gnode->get_op_ptr());
    NNFUSION_CHECK_NOT_NULLPTR(op);
    m_loop_body_tu = op->get_loop_body_tu();
    size_t workspace_size = 0;
    for (auto& pair : m_loop_body_tu->memory_allocator_factory->get_allocator_list())
    {
        workspace_size += pair.second->max_allocated();
    }
    m_workspace = allocate_tensor(Shape{workspace_size}, nnfusion::element::character);
    m_context->inputs.push_back(m_workspace);
    m_context->input_names.push_back(m_workspace->get_name());
    m_loop_output_map = op->get_loop_output_map();
    m_shared_memory_size = get_subgraph_shared_memory(m_loop_body_tu->program);
    bypass_instructions(m_loop_body_tu->program);
}

void cuda::Loop::generate_subgraph_code(LanguageUnit_p _lu)
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
            {
                auto input_index = inputs[tensor->get_name()];
                if (input_index == -1)
                    params.push_back("&i");
                else
                    params.push_back("input" + std::to_string(input_index));
            }
            else
                params.push_back(get_workspace_tensor(tensor));
        }
        for (auto tensor : ins->get_outputs())
        {
            if (m_loop_output_map.count(tensor->get_name(false)))
            {
                auto output_index = m_loop_output_map[tensor->get_name(false)];
                if (output_index == 0)
                {
                    params.push_back("input1");
                }
                else
                {
                    params.push_back("output" + std::to_string(output_index - 1));
                }
            }
            else
                params.push_back(get_workspace_tensor(tensor));
        }
        if (std::dynamic_pointer_cast<BlockFusionCudaCodegen>(kernel) != nullptr)
            for (auto tensor : kernel->m_context->tensors)
                params.push_back(get_workspace_tensor(tensor));
        lu << kernel->emit_block_kernel_call(params)->get_code();
        lu << "Barrier();\n";
    }
}

LanguageUnit_p cuda::Loop::emit_function_body()
{
    LanguageUnit_p _lu(new LanguageUnit(get_function_name()));
    auto& lu = *_lu;
    allocate_shared_memory(_lu);
    lu << "for (int64_t i = 0; i < *input0; i++)";
    lu.block_begin();
    // after the first loop, loop-carried output should be used as input
    lu << "if (i == 1)";
    lu.block_begin();
    for (int i = 0; i < m_context->outputs.size(); i++)
        lu << "input" << i + 2 << " = output" << i << ";\n";
    lu.block_end();
    generate_subgraph_code(_lu);
    lu.block_end();
    return _lu;
}

void cuda::Loop::set_launch_config()
{
    auto cfg0 = get_subgraph_launch_config(m_loop_body_tu->program);
    m_blockDim = cfg0.first;
    m_gridDim = cfg0.second;
}

LanguageUnit_p cuda::Loop::emit_dependency()
{
    LanguageUnit_p _lu(new LanguageUnit(get_function_name() + "_dep"));
    _lu->require(header::cuda);
    _lu->require(declaration::barrier);
    for (auto ins : get_fused_kernel(m_loop_body_tu->program))
    {
        auto kernel = static_pointer_cast<cuda::CudaEmitter>(ins->getKernel());
        auto body = kernel->get_or_emit_source();
        auto block_kernel = kernel->emit_block_kernel();
        block_kernel->require(body->dep_unit);
        _lu->require(block_kernel);
    }
    return _lu;
}

REGISTER_KERNEL_EMITTER("Loop",                                                    // op_name
                        Device(CUDA_GPU).TypeConstraint(element::f32).Priority(2), // attrs
                        cuda::Loop)

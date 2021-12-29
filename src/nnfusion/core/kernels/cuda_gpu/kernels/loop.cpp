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
        m_pool_offset[pair.second->get_name()] = workspace_size;
        workspace_size += pair.second->max_allocated();
    }
    m_workspace = allocate_tensor(Shape{workspace_size}, nnfusion::element::character);
    m_context->inputs.push_back(m_workspace);
    m_context->input_names.push_back(m_workspace->get_name());
    auto output_map = op->get_loop_output_map();
    m_shared_memory_size = get_subgraph_shared_memory(m_loop_body_tu->program);
    for (auto& item : output_map)
        item.second--;
    m_body_instructions = create_param_map(m_loop_body_tu->program, output_map);
}

void cuda::Loop::generate_subgraph_code(LanguageUnit_p _lu)
{
    auto& lu = *_lu;
    for (auto ins : *m_body_instructions)
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
        lu << "Barrier();\n";
    }
}

LanguageUnit_p cuda::Loop::emit_function_body()
{
    LanguageUnit_p _lu(new LanguageUnit(get_function_name()));
    auto& lu = *_lu;
    allocate_shared_memory(_lu);
    lu << "int tid=threadIdx.x + blockIdx.x * blockDim.x;\n";
    for (int i = 0; i < m_context->outputs.size(); i++)
    {
        size_t tensor_size = m_context->outputs[i]->size(false);
        size_t num_threads = m_blockDim.x * m_gridDim.x;
        lu << "for (int64_t i=tid; i<" << tensor_size << "; i+=" << num_threads << ")";
        lu << " output" << i << "[i] = input" << i + 2 << "[i];\n";
    }
    for (int i = 0; i < m_context->outputs.size(); i++)
        lu << "input" << i + 2 << " = output" << i << ";\n";
    lu << "Barrier();\n";
    lu << "for (int64_t i = 0; i < *input0; i++)";
    lu.block_begin();
    generate_subgraph_code(_lu);
    lu.block_end();
    return _lu;
}

void cuda::Loop::set_launch_config()
{
    auto cfg0 = get_subgraph_launch_config(m_body_instructions);
    m_blockDim = cfg0.first;
    m_gridDim = cfg0.second;
}

LanguageUnit_p cuda::Loop::emit_dependency()
{
    LanguageUnit_p _lu(new LanguageUnit(get_function_name() + "_dep"));
    _lu->require(header::cuda);
    _lu->require(declaration::barrier);
    for (auto ins : *m_body_instructions)
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

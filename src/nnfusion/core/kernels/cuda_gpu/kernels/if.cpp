// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "if.hpp"
#include "../cuda_cudnn.hpp"
#include "convolution.hpp"
#include "nnfusion/core/operators/op_define/if.hpp"
#include "nnfusion/engine/pass/graph/blockfusion/blockfusion_codegen.hpp"

using namespace nnfusion;
using namespace nnfusion::kernels;

static inline cuda::dim3 maxdim3(cuda::dim3 lhs, cuda::dim3 rhs)
{
    return cuda::dim3(std::max(lhs.x, rhs.x), std::max(lhs.y, rhs.y), std::max(lhs.z, rhs.z));
}

static std::pair<cuda::dim3, cuda::dim3> get_subgraph_launch_config(const ir::Program& program)
{
    cuda::dim3 block_dim{0, 0, 0}, grid_dim{0, 0, 0};
    for (auto blk : program)
    {
        for (auto ins : *blk)
        {
            auto kernel = ins->getKernel();
            // neglect the Constant copy
            if (kernel == nullptr || ins->getGNode()->get_op_type() == "Result" ||
                ins->getGNode()->get_op_type() == "Constant")
                continue;
            if (kernel->get_kernel_type() == "cuda")
            {
                auto cuda_kernel = static_pointer_cast<cuda::CudaEmitter>(kernel);
                block_dim = maxdim3(block_dim, cuda_kernel->get_block_dim());
                grid_dim = maxdim3(grid_dim, cuda_kernel->get_grid_dim());
            }
            else
            {
                std::cout << kernel->get_kernel_type() << " " << ins->getGNode()->get_op_type()
                          << std::endl;
                NNFUSION_CHECK_FAIL();
            }
        }
    }
    return std::make_pair(block_dim, grid_dim);
}

static std::map<std::string, int> get_subgraph_inputs(const ir::Program& program)
{
    std::map<std::string, int> inputs;
    int i = 0;
    for (auto blk : program)
        for (auto ins : *blk)
        {
            if (ins->getGNode()->get_op_type() == "Parameter" ||
                ins->getGNode()->get_op_type() == "Constant")
            {
                auto input_map = (*ins->getGNode())["subgraph_input_map"];
                NNFUSION_CHECK(input_map.is_valid());
                inputs[ins->get_outputs()[0]->get_name()] = input_map.as<int>();
            }
        }
    return inputs;
}

static std::map<std::string, int> get_subgraph_outputs(graph::Graph::Pointer graph)
{
    std::map<std::string, int> inputs;
    int i = 0;
    for (auto node : graph->get_outputs())
        inputs[node->get_input_tensor(0).get_name()] = i++;
    return inputs;
}

static std::vector<ir::Instruction::Pointer> get_fused_kernel(const ir::Program& program)
{
    std::vector<ir::Instruction::Pointer> result;
    for (auto blk : program)
    {
        for (auto ins : *blk)
        {
            auto kernel = ins->getKernel();
            // neglect the Constant copy
            if (kernel == nullptr || ins->getGNode()->get_op_type() == "Result" ||
                ins->getGNode()->get_op_type() == "Constant")
                continue;
            if (kernel->get_kernel_type() == "cuda")
                result.push_back(ins);
            else if (kernel->get_kernel_type() == "cuda_lib")
            {
                auto op = ins->getGNode()->get_op_ptr();
                for (auto input : ins->get_inputs())
                {
                    std::cout << input->get_shape() << std::endl;
                }
                std::cout << ins->getGNode()->get_op_type() << "\n";
            }
        }
    }
    return result;
}

cuda::If::If(shared_ptr<KernelContext> ctx)
    : CudaEmitter(ctx)
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

std::string cuda::If::get_workspace_tensor(nnfusion::descriptor::Tensor::Pointer tensor)
{
    auto type = tensor->get_element_type().c_type_string();
    size_t offset = tensor->get_pool_offset();
    return "(" + type + "*)(input" + std::to_string(m_context->inputs.size() - 1) + "+" +
           std::to_string(offset) + ")";
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
        lu << "if (blockIdx.x < " << kernel->get_grid_dim().x << ")\n";
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
        for (auto tensor : kernel->m_context->tensors)
            params.push_back(get_workspace_tensor(tensor));
        lu << kernel->emit_block_kernel_call(params)->get_code();
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

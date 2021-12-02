// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "loop.hpp"
#include "../cuda_cudnn.hpp"
#include "convolution.hpp"
#include "nnfusion/core/operators/op_define/loop.hpp"

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

cuda::Loop::Loop(shared_ptr<KernelContext> ctx)
    : BlockCudaEmitter(ctx)
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
}

std::string cuda::Loop::get_workspace_tensor(nnfusion::descriptor::Tensor::Pointer tensor)
{
    auto type = tensor->get_element_type().c_type_string();
    size_t offset = tensor->get_pool_offset();
    return "(" + type + "*)(input" + std::to_string(m_context->inputs.size() - 1) + "+" +
           std::to_string(offset) + ")";
}

void cuda::Loop::generate_subgraph_code(LanguageUnit_p _lu)
{
    auto& lu = *_lu;
    auto instructions = get_fused_kernel(m_loop_body_tu->program);
    auto inputs = get_subgraph_inputs(m_loop_body_tu->program);
    for (auto ins : instructions)
    {
        std::vector<string> params;
        for (auto tensor : ins->get_inputs())
        {
            if (inputs.count(tensor->get_name()))
            {
                auto input_index = inputs[tensor->get_name()];
                if (input_index == 0)
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
        auto kernel = static_pointer_cast<cuda::CudaEmitter>(ins->getKernel());
        if (kernel->type() == "BlockFusionCudaCodegen")
            for (auto tensor : kernel->m_context->tensors)
                params.push_back(get_workspace_tensor(tensor));
        lu << kernel->emit_block_kernel_call(params)->get_code();
    }
}

LanguageUnit_p cuda::Loop::emit_function_body()
{
    LanguageUnit_p _lu(new LanguageUnit(get_function_name()));
    auto& lu = *_lu;
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

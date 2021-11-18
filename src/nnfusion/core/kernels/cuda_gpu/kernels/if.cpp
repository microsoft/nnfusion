// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "if.hpp"
#include "../cuda_cudnn.hpp"
#include "convolution.hpp"
#include "nnfusion/core/operators/op_define/if.hpp"
#include "nnfusion/engine/pass/graph/blockfusion/blockfusion_codegen.hpp"

using namespace nnfusion;
using namespace nnfusion::kernels;

static std::pair<cuda::dim3, cuda::dim3> get_subgraph_launch_config(const ir::Program& program)
{
    for (auto blk : program)
    {
        for (auto ins : *blk)
        {
            auto kernel = ins->getKernel();
            // neglect the Constant copy
            if (kernel == nullptr || ins->getGNode()->get_op_type() == "Constant")
                continue;
            if (kernel->get_kernel_type() == "cuda")
            {
                auto cuda_kernel = static_pointer_cast<cuda::CudaEmitter>(kernel);
                return std::make_pair(cuda_kernel->get_block_dim(), cuda_kernel->get_grid_dim());
            }
        }
    }
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
                inputs[ins->get_outputs()[0]->get_name()] =
                    ins->getGNode()->Get<int>("subgraph_input_map");
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

static ir::Instruction::Pointer get_fused_kernel(const ir::Program& program)
{
    for (auto blk : program)
    {
        for (auto ins : *blk)
        {
            auto kernel = ins->getKernel();
            // neglect the Constant copy
            if (kernel == nullptr || ins->getGNode()->get_op_type() == "Constant")
                continue;
            if (kernel->get_kernel_type() == "cuda")
                return ins;
        }
    }
    return nullptr;
}

static bool check_subgraph_isvalid(const ir::Program& program)
{
    std::unordered_set<ir::Instruction::Pointer> set;
    for (auto blk : program)
    {
        for (auto ins : *blk)
        {
            auto kernel = ins->getKernel();
            // neglect the Constant copy
            if (kernel == nullptr || ins->getGNode()->get_op_type() == "Constant")
                continue;
            if (kernel->get_kernel_type() == "cuda")
            {
                set.insert(ins);
            }
        }
    }
    if (set.size() > 1)
    {
        for (auto p : set)
        {
            std::cout << p->getGNode()->get_op_type() << std::endl;
        }
    }
    return set.size() <= 1;
}

static bool dim3_is_equal(const cuda::dim3& lhs, const cuda::dim3& rhs)
{
    return lhs.x == rhs.x && lhs.y == rhs.y && lhs.z == rhs.z;
}

cuda::If::If(shared_ptr<KernelContext> ctx)
    : BlockCudaEmitter(ctx)
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
}

void cuda::If::generate_branch_code(LanguageUnit_p _lu, bool else_branch = false)
{
    auto tu = m_then_branch_tu;
    if (else_branch)
    {
        tu = m_else_branch_tu;
    }
    auto& lu = *_lu;
    auto ins = get_fused_kernel(tu->program);
    if (ins == nullptr)
        return;
    auto inputs = get_subgraph_inputs(tu->program);
    auto outputs = get_subgraph_outputs(tu->m_graph);
    NNFUSION_CHECK(m_context->outputs.size() == outputs.size());
    std::vector<string> params;
    int tensor_cnt = 0;
    for (auto tensor : ins->get_inputs())
    {
        NNFUSION_CHECK(inputs.count(tensor->get_name()));
        params.push_back("input" + std::to_string(inputs[tensor->get_name()] - 1));
    }
    for (auto tensor : ins->get_outputs())
    {
        if (outputs.count(tensor->get_name()))
            params.push_back("output" + std::to_string(outputs[tensor->get_name()]));
        else
        {
            params.push_back("temp" + std::to_string(tensor_cnt));
            auto type = tensor->get_element_type().c_type_string();
            size_t offset = tensor->get_pool_offset();
            lu << type << "* temp" << tensor_cnt++ << " = (" << type << "*)(workspace + " << offset
               << ");\n";
        }
    }
    auto kernel = static_pointer_cast<cuda::CudaEmitter>(ins->getKernel());
    for (auto tensor : kernel->m_context->tensors)
    {
        params.push_back("temp" + std::to_string(tensor_cnt));
        auto type = tensor->get_element_type().c_type_string();
        size_t offset = tensor->get_pool_offset();
        lu << type << "* temp" << tensor_cnt++ << " = (" << type << "*)(workspace + " << offset
           << ");\n";
    }
    lu << kernel->emit_block_kernel_call(params)->get_code();
    auto body = kernel->get_or_emit_source();
    lu.require(body->dep_unit);
    lu.require(kernel->emit_block_kernel());
}

LanguageUnit_p cuda::If::emit_function_body()
{
    LanguageUnit_p _lu(new LanguageUnit(get_function_name()));
    auto& lu = *_lu;
    lu << "if (*cond) ";
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
    NNFUSION_CHECK(check_subgraph_isvalid(m_then_branch_tu->program)) << "then-branch is invalid";
    NNFUSION_CHECK(check_subgraph_isvalid(m_else_branch_tu->program)) << "else-branch is invalid";
    auto cfg0 = get_subgraph_launch_config(m_then_branch_tu->program);
    auto cfg1 = get_subgraph_launch_config(m_then_branch_tu->program);
    NNFUSION_CHECK(dim3_is_equal(cfg0.first, cfg1.first) && dim3_is_equal(cfg0.second, cfg1.second))
        << "then-branch and else-branch must have equal grid and blocks";
    m_blockDim = cfg0.first;
    m_gridDim = cfg0.second;
}

LanguageUnit_p cuda::If::emit_dependency()
{
    LanguageUnit_p _lu(new LanguageUnit(get_function_name() + "_dep"));
    _lu->require(header::cuda);
    return _lu;
}

// rename the first input with cond
// other inputs starts from 1...
LanguageUnit_p cuda::If::emit_function_signature()
{
    LanguageUnit_p _lu(new LanguageUnit(this->m_kernel_name + "_sig"));
    auto& lu = *_lu;

    vector<string> params;
    for (size_t i = 0; i < m_context->inputs.size(); i++)
    {
        stringstream ss;
        ss << m_context->inputs[i]->get_element_type().c_type_string() << "* ";
        if (i == 0)
            ss << "cond";
        else if (i == m_context->inputs.size() - 1)
            ss << "workspace";
        else
            ss << "input" << i - 1;
        params.push_back(ss.str());
    }

    for (size_t i = 0; i < m_context->outputs.size(); i++)
    {
        stringstream ss;
        ss << m_context->outputs[i]->get_element_type().c_type_string() << "* ";
        ss << "output" << i;
        params.push_back(ss.str());
    }

    lu << "extern \"C\" __launch_bounds__(" << m_blockDim.x * m_blockDim.y * m_blockDim.z
       << ") __global__ void "
       << "(" << join(params, ", ") << ")";
    return _lu;
}

REGISTER_KERNEL_EMITTER("If",                                                      // op_name
                        Device(CUDA_GPU).TypeConstraint(element::f32).Priority(2), // attrs
                        cuda::If)

// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "recursion.hpp"
#include "../cuda_cudnn.hpp"
#include "convolution.hpp"
#include "nnfusion/core/operators/op_define/recursion.hpp"

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
                // NNFUSION_CHECK_FAIL();
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
            if (ins->getGNode()->get_op_type() == "Parameter")
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

cuda::FuncForward::FuncForward(shared_ptr<KernelContext> ctx)
    : CudaEmitter(ctx)
{
    std::stringstream tag;
    tag << "_FuncForwardOP";
    custom_tag = tag.str();
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

    lu << m_block_func_name + "_block_kernel(" << join(params, ", ") << ");\n";
    return _lu;
}

std::string cuda::FuncForward::m_block_func_name = "";

cuda::Recursion::Recursion(shared_ptr<KernelContext> ctx)
    : CudaEmitter(ctx)
{
    std::stringstream tag;
    tag << "_RecursionOP";
    custom_tag = tag.str();
    auto op = static_pointer_cast<op::Recursion>(ctx->gnode->get_op_ptr());
    NNFUSION_CHECK_NOT_NULLPTR(op);
    m_loop_body_tu = op->get_body_tu();
    size_t workspace_size = 0;
    for (auto& pair : m_loop_body_tu->memory_allocator_factory->get_allocator_list())
    {
        workspace_size += pair.second->max_allocated();
    }
    m_workspace = allocate_tensor(Shape{workspace_size}, nnfusion::element::character);
    m_context->inputs.push_back(m_workspace);
    m_context->input_names.push_back(m_workspace->get_name());
    m_loop_output_map = op->get_output_map();
    m_block_func_name = move(FuncForward::m_block_func_name);
    NNFUSION_CHECK(!m_block_func_name.empty());
}

std::string cuda::Recursion::get_workspace_tensor(nnfusion::descriptor::Tensor::Pointer tensor)
{
    auto type = tensor->get_element_type().c_type_string();
    size_t offset = tensor->get_pool_offset();
    return "(" + type + "*)(input" + std::to_string(m_context->inputs.size() - 1) + "+" +
           std::to_string(offset) + ")";
}

void cuda::Recursion::generate_subgraph_code(LanguageUnit_p _lu)
{
    auto& lu = *_lu;
    auto instructions = get_fused_kernel(m_loop_body_tu->program);
    auto inputs = get_subgraph_inputs(m_loop_body_tu->program);
    for (auto ins : instructions)
    {
        auto kernel = static_pointer_cast<cuda::CudaEmitter>(ins->getKernel());
        lu << "if (blockIdx.x < " << kernel->get_grid_dim().x << ")\n";
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
        for (auto tensor : kernel->m_context->tensors)
            params.push_back(get_workspace_tensor(tensor));
        lu << kernel->emit_block_kernel_call(params)->get_code();
    }
}

LanguageUnit_p cuda::Recursion::emit_function_body()
{
    LanguageUnit_p _lu(new LanguageUnit(get_function_name()));
    auto& lu = *_lu;
    generate_subgraph_code(_lu);
    m_saved_func_body = _lu;
    return _lu;
}

void cuda::Recursion::set_launch_config()
{
    auto cfg0 = get_subgraph_launch_config(m_loop_body_tu->program);
    m_blockDim = cfg0.first;
    m_gridDim = cfg0.second;
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
    (*kernel_code) << m_saved_func_body->get_code();
    kernel_code->block_end();
    m_kernel_name = saved;

    for (auto ins : get_fused_kernel(m_loop_body_tu->program))
    {
        auto kernel = static_pointer_cast<cuda::CudaEmitter>(ins->getKernel());
        auto body = kernel->get_or_emit_source();
        auto block_kernel = kernel->emit_block_kernel();
        block_kernel->require(body->dep_unit);
        _lu->require(block_kernel);
    }

    _lu->require(kernel_code);

    return _lu;
}

REGISTER_KERNEL_EMITTER("FuncForward",                                             // op_name
                        Device(CUDA_GPU).TypeConstraint(element::f32).Priority(2), // attrs
                        cuda::FuncForward)
REGISTER_KERNEL_EMITTER("Recursion",                                               // op_name
                        Device(CUDA_GPU).TypeConstraint(element::f32).Priority(2), // attrs
                        cuda::Recursion)

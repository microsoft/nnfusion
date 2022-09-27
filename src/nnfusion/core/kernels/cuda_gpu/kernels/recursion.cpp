// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "recursion.hpp"
#include "../cuda_cudnn.hpp"
#include "convolution.hpp"
#include "nnfusion/core/operators/op_define/recursion.hpp"
#include "nnfusion/engine/pass/graph/blockfusion/blockfusion_codegen.hpp"

using namespace nnfusion;
using namespace nnfusion::kernels;

DEFINE_bool(frecursive_stack, false, "recursive with manual stack");

cuda::FuncForward::FuncForward(shared_ptr<KernelContext> ctx)
    : CudaEmitter(ctx)
{
    std::stringstream tag;
    tag << "_FuncForwardOP";
    custom_tag = tag.str();
    m_workspace = allocate_tensor(Shape{0}, nnfusion::element::character);
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

void cuda::FuncForward::update_context_from_gnode(std::shared_ptr<nnfusion::graph::GNode> gnode) {
    auto ctx = std::make_shared<KernelContext>(gnode);
    m_context = ctx;
    m_context->inputs.push_back(m_workspace);
    m_context->input_names.push_back(m_workspace->get_name());
}

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
        m_pool_offset[pair.second->get_name()] = m_workspace_size;
        m_workspace_size += pair.second->max_allocated();
    }
    m_workspace = allocate_tensor(Shape{m_workspace_size * 20}, nnfusion::element::character);
    m_context->inputs.push_back(m_workspace);
    m_context->input_names.push_back(m_workspace->get_name());
    m_block_func_name = move(FuncForward::m_block_func_name);
    NNFUSION_CHECK(!m_block_func_name.empty());
    m_shared_memory_size = get_subgraph_shared_memory(m_loop_body_tu->program);
    m_body_instructions = create_param_map(m_loop_body_tu->program, op->get_output_map(), true);
}

std::string cuda::Recursion::inline_kernel(std::shared_ptr<ir::Instruction> ins) {
    auto cuda_kernel = dynamic_pointer_cast<CudaEmitter>(ins->getKernel());
    std::string code = cuda_kernel->emit_device_function_body()->get_code();
    code = replace_all(code, "input", "IN_PUT_");
    code = replace_all(code, "output", "OUT_PUT_");
    auto inputs = ins->get_inputs();
    for (int i = 0; i < inputs.size(); i++) {
        stringstream ss_from;
        ss_from << "IN_PUT_" << i;
        std::string from = ss_from.str();
        NNFUSION_LOG(INFO) << "replace " << from << " to " << m_param_map[inputs[i]];
        code = replace_all(code, from, "(" + m_param_map[inputs[i]] + ")");
    }
    auto outputs = ins->get_outputs();
    for (int i = 0; i < outputs.size(); i++) {
        stringstream ss_from;
        ss_from << "OUT_PUT_" << i;
        std::string from = ss_from.str();
        NNFUSION_LOG(INFO) << "replace " << from << " to " << m_param_map[outputs[i]];
        code = replace_all(code, from, "(" + m_param_map[outputs[i]] + ")");
    }
    return code;
}

void cuda::Recursion::generate_subgraph_code(LanguageUnit_p _lu)
{
    auto& lu = *_lu;
    auto instructions = m_body_instructions;
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
        if (FLAGS_frecursive_stack && is_emitting_block_kernel) {
            lu << "// " << kernel->emit_block_kernel_call(params)->get_code();
            lu << inline_kernel(ins);
        } else {
            lu << kernel->emit_block_kernel_call(params)->get_code();
        }
        if (ins != instructions->back())
            lu << "Barrier();\n";
    }
}

LanguageUnit_p cuda::Recursion::emit_function_body()
{
    LanguageUnit_p _lu(new LanguageUnit(get_function_name()));
    auto& lu = *_lu;
    allocate_shared_memory(_lu);
    if (FLAGS_frecursive_stack && !is_emitting_block_kernel) {
        std::vector<std::string> params;
        for (size_t i = 0; i < m_context->inputs.size(); i++) {
            stringstream ss;
            ss << "input" << i;
            params.push_back(ss.str());
        }
        for (size_t i = 0; i < m_context->outputs.size(); i++) {
            stringstream ss;
            ss << "output" << i;
            params.push_back(ss.str());
        }
        params.push_back("shared_buffer");
        lu << m_block_func_name << "_block_kernel(" << join(params, ", ") << ");";
    } else {
        generate_subgraph_code(_lu);
    }
    return _lu;
}

void cuda::Recursion::set_launch_config()
{
    auto cfg0 = get_subgraph_launch_config(m_body_instructions);
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

    for (auto ins : *m_body_instructions)
    {
        auto kernel = static_pointer_cast<cuda::CudaEmitter>(ins->getKernel());
        auto body = kernel->get_or_emit_source();
        auto block_kernel = kernel->emit_block_kernel();
        block_kernel->require(body->dep_unit);
        // HACK: still need to emit the if kernel when inlined, for emiting the kernels the ``if'' depends on
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
